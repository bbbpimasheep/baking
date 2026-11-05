from dataclasses import replace
import csv
import json
import numpy as np
import pytest
import warp as wp
from config import DingConfig, ExperimentConfig, LoadExperimentConfig, MaterialConfig, SimulationConfig
from experiments.recording import SimulationRecorder
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from geometry.mesh_validation import ValidateTetMesh
from physics.materials.ding import DingMaterial, porosity_coefficient
from physics.mpm import MpmSimulator
from physics.state import AllocateState


def MakeThermalPoint(ding_config, requires_grad=False, initial_temperature_k=293.15):
    material = DingMaterial(MaterialConfig(), ding_config, 'cpu', requires_grad, initial_temperature_k=initial_temperature_k)
    state = AllocateState(1, 'cpu', requires_grad)
    state.deformation_gradients.assign(np.eye(3, dtype=np.float32)[None])
    material.InitializeState(state)
    return material, state


def AdvancePoint(material, old_state, total_gradient, time_value=0.01, time_step=0.01):
    state = AllocateState(1, 'cpu', old_state.particle_positions.requires_grad)
    increment = total_gradient @ np.linalg.inv(old_state.deformation_gradients.numpy()[0])
    state.deformation_gradients.assign(total_gradient.astype(np.float32)[None])
    state.affine_velocities.assign(((increment - np.eye(3)) / time_step).astype(np.float32)[None])
    material.AdvanceState(old_state, state, time_step)
    material.CheckState()
    return state


@wp.kernel
def EvaluateCoefficients(porosities: wp.array(dtype=float), coefficients: wp.array(dtype=float)):
    particle_index = wp.tid()
    coefficients[particle_index] = porosity_coefficient(porosities[particle_index])


def TestRisingPenaltyViscosityDuringSetting():
    config = DingConfig(temperature_enabled=True, yield_stress=500.0, viscosity=1e9,
                        temperature_knots_k=(278.15, 293.15, 328.15, 353.15),
                        yield_factors=(1.5, 1.0, 0.5, 3.0), viscosity_factors=(1.5, 1.0, 0.5, 5.0))
    config.Validate()
    # Cover initialization and return-map updates, including the rising segment.
    for temperature, factor in ((328.15, 0.5), (340.65, 2.75), (353.15, 5.0)):
        material, state = MakeThermalPoint(config, initial_temperature_k=temperature)
        for checked in (state, AdvancePoint(material, state, np.eye(3))):
            np.testing.assert_allclose(checked.material_state.viscosities.numpy(), config.viscosity * factor, rtol=2e-6)
    # Under the same super-yield loading, increased penalty viscosity retains more stress.
    stress_norms = []
    for hot_factor in (0.5, 5.0):
        material, state = MakeThermalPoint(replace(config, viscosity_factors=(1.5, 1.0, 0.5, hot_factor)), initial_temperature_k=353.15)
        state = AdvancePoint(material, state, np.diag([1.2, 0.9, 1.04]))
        assert np.isfinite(state.material_state.kirchhoff_stresses.numpy()).all()
        stress_norms.append(np.linalg.norm(state.material_state.kirchhoff_stresses.numpy()))
    assert stress_norms[1] > stress_norms[0]


def TestTemperatureAndPorosityParameterValues():
    config = DingConfig(temperature_enabled=True)
    # Test both host initialization and kernel updates at knots, in segments and outside.
    for temperature_value, yield_factor, viscosity_factor in (
            (260.0, 1.5, 1.5), (278.15, 1.5, 1.5), (293.15, 1.0, 1.0),
            (303.15, 0.6, 0.6), (313.15, 0.2, 0.2), (333.15, 1.1, 0.125),
            (353.15, 2.0, 0.05), (400.0, 2.0, 0.05)):
        material, state = MakeThermalPoint(config, initial_temperature_k=temperature_value)
        for checked_state in (state, AdvancePoint(material, state, np.eye(3))):
            internal_state = checked_state.material_state
            np.testing.assert_allclose(internal_state.yield_stresses.numpy(), config.yield_stress * yield_factor, rtol=2e-6)
            np.testing.assert_allclose(internal_state.viscosities.numpy(), config.viscosity * viscosity_factor, rtol=2e-6)
            np.testing.assert_allclose(internal_state.temperatures.numpy(), temperature_value, atol=2e-5)
    porosities = np.array([0.0, 0.01, 0.1, 0.2, 0.5, 0.999], dtype=np.float32)
    coefficients = wp.zeros(len(porosities), dtype=float, device='cpu')
    wp.launch(EvaluateCoefficients, len(porosities), inputs=[wp.array(porosities, device='cpu'), coefficients], device='cpu')
    expected_values = [0.0] + [9 / (4 * np.log(1 / value) ** 2 * (3.5 * value + 1)) for value in porosities[1:]]
    np.testing.assert_allclose(coefficients.numpy(), expected_values, rtol=5e-5)
    assert np.all(np.diff(coefficients.numpy()) > 0)
    print('temperature table checked; a(n):', coefficients.numpy().tolist())


def TestInitialPorosityAndElasticVersusPlasticVolume():
    for porosity in (0.0, 0.2, 0.5):
        material, state = MakeThermalPoint(DingConfig(yield_stress=1e9, porosity=porosity))
        np.testing.assert_allclose(state.material_state.porosities.numpy(), porosity)
        # Purely elastic volume change must not drive porosity.
        state = AdvancePoint(material, state, np.diag([1.1, 0.95, 1.02]))
        np.testing.assert_allclose(state.material_state.porosities.numpy(), porosity, atol=1e-6)
    for stretch_value in (0.98, 1.02):
        material, state = MakeThermalPoint(DingConfig(yield_stress=100.0, porosity=0.2))
        for step_index in range(5):
            state = AdvancePoint(material, state, np.eye(3) * stretch_value, (step_index + 1) * 0.01)
            plastic_volume = np.linalg.det(state.material_state.plastic_gradients.numpy())
            np.testing.assert_allclose(state.material_state.porosities.numpy(), plastic_volume - 1 + 0.2, atol=2e-7)
        final_porosity = state.material_state.porosities.numpy()[0]
        assert (final_porosity - 0.2) * (stretch_value - 1) > 0
        print('hydrostatic stretch:', stretch_value, 'final porosity:', final_porosity)


def TestUpdatedPorosityDefinesNextReturnMap():
    config = DingConfig(yield_stress=100.0, porosity=0.2)
    material, state = MakeThermalPoint(config)
    total_gradient = np.eye(3) * 0.98
    state = AdvancePoint(material, state, total_gradient)
    old_stress = state.material_state.kirchhoff_stresses.numpy()[0].copy()
    old_porosity = float(state.material_state.porosities.numpy()[0])
    state = AdvancePoint(material, state, total_gradient)
    tau_values = np.linalg.eigvalsh(state.material_state.kirchhoff_stresses.numpy()[0]).astype(float)
    trial_values = np.linalg.eigvalsh(old_stress).astype(float)
    shear_modulus = material.material_config.shear_modulus
    lame_lambda = material.material_config.lame_lambda
    compliance = np.linalg.inv(2 * shear_modulus * np.eye(3) + lame_lambda * np.ones((3, 3)))
    coefficient = 9 / (4 * np.log(1 / old_porosity) ** 2 * (3.5 * old_porosity + 1))
    mean_value = tau_values.mean()
    deviator = tau_values - mean_value
    yield_value = 1.5 * np.dot(deviator, deviator) + coefficient * mean_value ** 2 - config.yield_stress ** 2
    yield_derivative = 3 * deviator + (2 * coefficient / 3) * mean_value
    residual_values = compliance @ (tau_values - trial_values) + 0.01 / config.viscosity * max(yield_value, 0) * yield_derivative
    np.testing.assert_allclose(residual_values, 0, atol=2e-6)
    assert abs(old_porosity - config.porosity) > 1e-4


def TestHeatingChangesResponseWithoutThermalExpansion():
    stress_norms = []
    for temperature_value in (293.15, 313.15, 353.15):
        config = DingConfig(temperature_enabled=True,
                            yield_stress=500.0, porosity=0.2)
        material, state = MakeThermalPoint(config, initial_temperature_k=temperature_value)
        for step_index in range(20):
            state = AdvancePoint(material, state, np.diag([1.05, 1 / np.sqrt(1.05), 1 / np.sqrt(1.05)]), (step_index + 1) * 0.01)
        stresses = state.material_state.kirchhoff_stresses.numpy()[0]
        stress_norms.append(float(np.linalg.norm(stresses - np.trace(stresses) / 3 * np.eye(3))))
    assert stress_norms[1] < stress_norms[0] < stress_norms[2]
    print('room/melt/gel deviatoric stress norms, Pa:', stress_norms)
    material, state = MakeThermalPoint(DingConfig(temperature_enabled=True, porosity=0.2))
    for step_index in range(100):
        state = AdvancePoint(material, state, np.eye(3), (step_index + 1) * 0.025, 0.025)
    np.testing.assert_allclose(state.material_state.elastic_gradients.numpy(), np.eye(3, dtype=np.float32)[None], atol=2e-6)
    np.testing.assert_allclose(state.material_state.kirchhoff_stresses.numpy(), 0.0, atol=0.05)
    np.testing.assert_allclose(state.material_state.porosities.numpy(), 0.2, atol=2e-6)


@pytest.mark.parametrize('invalid_values', [
    dict(temperature_knots_k=(278, 293, 290, 353)), dict(viscosity_factors=(1, float('inf'), 0.2, 0.1)),
    dict(viscosity_factors=(1, 0.5, 0.2, 0)), dict(yield_factors=(1, 2, 0.2, 3)),
    dict(porosity=-0.1), dict(porosity=1.0)])
def TestTemperatureAndPorosityDomains(invalid_values):
    with pytest.raises(ValueError):
        replace(DingConfig(), **invalid_values).Validate()


@pytest.mark.parametrize('porosity,stretch_value', [(0.95, 1.2), (0.2, 0.7)])
def TestPorosityExceedingDomainIsReportedWithoutClamping(porosity, stretch_value):
    material, state = MakeThermalPoint(DingConfig(porosity=porosity, yield_stress=100.0, viscosity=1e4))
    with pytest.raises(RuntimeError, match='porosity left'):
        AdvancePoint(material, state, np.eye(3) * stretch_value)


@wp.kernel
def ThermalStressLoss(stresses: wp.array(dtype=wp.mat33), loss_value: wp.array(dtype=float)):
    particle_index = wp.tid()
    wp.atomic_add(loss_value, 0, stresses[particle_index][0, 0] / float(stresses.shape[0]))


def EvaluateThermalMpmGradient(viscosity_value, backward=False):
    simulator = BuildThermalSimulator(retain_history=True, viscosity_value=viscosity_value, short_curve=True)
    loss_value = wp.zeros(1, dtype=float, device='cpu', requires_grad=True)
    tape_context = wp.Tape()
    with tape_context:
        for step_index in range(5):
            simulator.Step()
        wp.launch(ThermalStressLoss, simulator.particle_data.particle_count,
                  inputs=[simulator.current_state.material_state.kirchhoff_stresses, loss_value], device='cpu')
    simulator.CheckState()
    gradient_value = None
    if backward:
        tape_context.backward(loss_value)
        gradient_value = float(simulator.material.viscosity_parameter.grad.numpy()[0])
    return float(loss_value.numpy()[0]), gradient_value


def TestHeatedPorousMpmViscosityGradient():
    _, gradient_value = EvaluateThermalMpmGradient(1000.0, True)
    upper_value, _ = EvaluateThermalMpmGradient(1010.0)
    lower_value, _ = EvaluateThermalMpmGradient(990.0)
    finite_difference = (upper_value - lower_value) / 20.0
    assert abs(finite_difference) > 1e-7
    np.testing.assert_allclose(gradient_value, finite_difference, rtol=0.03, atol=1e-7)
    print('heated porous MPM eta_base AD:', gradient_value, 'FD:', finite_difference,
          'relative difference:', abs(gradient_value / finite_difference - 1))


def TestInvalidPoreStepDoesNotReplaceOrRecordTheLastValidState(tmp_path):
    simulator = BuildThermalSimulator(viscosity_value=1e4)
    positions = simulator.current_state.particle_positions.numpy()
    rates = -3000.0 * np.eye(3, dtype=np.float32)
    simulator.current_state.particle_velocities.assign((positions - np.array([0.5, 0.6, 0.5], dtype=np.float32)) @ rates.T)
    simulator.current_state.affine_velocities.assign(np.tile(rates, (len(positions), 1, 1)))
    config = replace(ExperimentConfig(), material=simulator.material.material_config, ding=simulator.material.ding_config)
    config = replace(config, run=replace(config.run, log_path=str(tmp_path / 'results')))
    recorder = SimulationRecorder(simulator, config)
    simulator.result_recorder = recorder
    initial_state = simulator.current_state
    with pytest.raises(RuntimeError, match='porosity left'):
        simulator.Step()
    assert simulator.current_state is initial_state
    assert simulator.step_count == 0 and len(recorder.expansion_history) == 1
    assert not (tmp_path / 'results').exists()



def BuildThermalSimulator(device_name='cpu', retain_history=False, viscosity_value=1e8, short_curve=False):
    from config import TransportConfig
    mesh = CreateDemoMesh(1)
    sim = MpmSimulator(mesh, SampleParticles(mesh, 1000., 4),
        replace(SimulationConfig(), device_name=device_name, time_step=.001 if short_curve else .0001,
                gravity=(0.,0.,0.)), MaterialConfig(model='ding2019'), retain_history=retain_history,
        ding_config=DingConfig(temperature_enabled=True, porosity=.2, yield_stress=5. if short_curve else 100., viscosity=viscosity_value),
        transport_config=TransportConfig(enabled=True, water_enabled=False, initial_temperature_k=300.))
    rates=np.diag([1.5,-.75,.2]).astype(np.float32)
    positions=sim.current_state.particle_positions.numpy()
    sim.current_state.particle_velocities.assign((positions-np.array([.5,.6,.5],dtype=np.float32))@rates.T)
    sim.current_state.affine_velocities.assign(np.tile(rates,(len(positions),1,1)))
    return sim


def TestGelatinizationFromParticleTemperaturePersistsOnCooling():
    material,state=MakeThermalPoint(DingConfig(temperature_enabled=True))
    for temperature, expected_yield, gelled in ((303.15,1200.,0),(313.15,400.,0),(353.15,4000.,1),(293.15,4000.,1)):
        state.material_state.temperatures.assign([temperature])
        state=AdvancePoint(material,state,np.eye(3))
        np.testing.assert_allclose(state.material_state.yield_stresses.numpy(),expected_yield,rtol=2e-6)
        assert state.material_state.gelled.numpy()[0]==gelled
