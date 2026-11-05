from dataclasses import replace
from pathlib import Path
import csv
import json
import math
import numpy as np
import pytest
import warp as wp
from config import DingConfig, ExperimentConfig, MaterialConfig, SimulationConfig, TransportConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.materials.ding import DingMaterial
from physics.materials.gas import gas_constant, gas_pressure
from physics.materials.porous_return import solve_porosity, porosity_prediction
from physics.mpm import MpmSimulator
from physics.state import AllocateState
from experiments.recording import SimulationRecorder


@wp.kernel
def PressureValues(moles: wp.array(dtype=wp.float64), temperatures: wp.array(dtype=float),
                   volumes: wp.array(dtype=float), porosities: wp.array(dtype=float), pressures: wp.array(dtype=wp.float64)):
    index = wp.tid()
    pressures[index] = gas_pressure(moles[index], temperatures[index], volumes[index], porosities[index])


def MakeGasPoint(config=None, requires_grad=False):
    config = config or DingConfig(gas_enabled=True, porosity=0.2)
    material = DingMaterial(MaterialConfig(), config, 'cpu', requires_grad)
    state = AllocateState(1, 'cpu', requires_grad)
    state.deformation_gradients.assign(np.eye(3, dtype=np.float32)[None])
    material.InitializeState(state)
    return material, state


def PointStep(material, state, gradient, time_value, time_step):
    output = AllocateState(1, 'cpu', state.particle_positions.requires_grad)
    increment = gradient @ np.linalg.inv(state.deformation_gradients.numpy()[0])
    output.deformation_gradients.assign(gradient.astype(np.float32)[None])
    output.affine_velocities.assign(((increment - np.eye(3)) / time_step).astype(np.float32)[None])
    material.AdvanceState(state, output, time_step)
    material.CheckState()
    return output


def TestIdealGasUnitsAndScaling():
    moles = wp.array([0.2] * 4, dtype=wp.float64, device='cpu')
    pressures = wp.zeros(4, dtype=wp.float64, device='cpu')
    wp.launch(PressureValues, 4, inputs=[moles, wp.array([300., 600., 300., 300.], dtype=float, device='cpu'),
        wp.array([1., 1., 2., 1.], dtype=float, device='cpu'), wp.array([0.2, 0.2, 0.2, 0.1], dtype=float, device='cpu'), pressures], device='cpu')
    np.testing.assert_allclose(pressures.numpy() / (gas_constant * 300), [1, 2, 0.5, 2], rtol=2e-7)


def TestStressConversionAmbientAndNoSourceConservation():
    material, state = MakeGasPoint(DingConfig(gas_enabled=True, porosity=0.2, gas_production_rate=0.0))
    stresses = wp.zeros(1, dtype=wp.mat33, device='cpu')
    material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    np.testing.assert_allclose(stresses.numpy(), 0, atol=0.02)
    state.material_state.gas_moles.assign(state.material_state.gas_moles.numpy() * 2)
    material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    np.testing.assert_allclose(stresses.numpy()[0], -101325 * np.eye(3), rtol=2e-6, atol=0.03)
    # At J=2 and vacuum exterior, Kirchhoff gas stress is still -NRT/n, not -NRT/(J*n).
    state.deformation_gradients.assign((np.eye(3, dtype=np.float32) * np.float32(2 ** (1 / 3)))[None])
    material.ding_config = replace(material.ding_config, gas_ambient_pressure_pa=0)
    material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    np.testing.assert_allclose(stresses.numpy()[0], -202650 * np.eye(3), rtol=2e-6, atol=0.05)
    state.deformation_gradients.assign(np.eye(3, dtype=np.float32)[None])
    initial_moles = state.material_state.gas_moles.numpy().copy()
    for index in range(10):
        state = PointStep(material, state, np.eye(3), (index + 1) * 0.01, 0.01)
    np.testing.assert_array_equal(state.material_state.gas_moles.numpy(), initial_moles)


def TestLocalGaussianSourceConstantTemperature():
    from physics.thermal_water import LocalGas
    for steps in (1,20,100):
        temperature=wp.array([333.15],dtype=wp.float64,device='cpu')
        old=wp.zeros(1,dtype=wp.float64,device='cpu')
        pressure=wp.zeros_like(old)
        gradients=wp.array([np.eye(3)],dtype=wp.mat33,device='cpu')
        porosity=wp.array([.2],dtype=float,device='cpu')
        parameter=wp.array([.002],dtype=float,device='cpu')
        for _ in range(steps):
            output=wp.zeros_like(old)
            wp.launch(LocalGas,1,inputs=[old,temperature,temperature,output,pressure,gradients,porosity,
                parameter,1000.,333.15,15.,.1/steps,1],device='cpu')
            old=output
        np.testing.assert_allclose(old.numpy(),.2,rtol=2e-6)


def TestCoupledCompactionConsistencyAndSmallPores():
    material, state = MakeGasPoint(DingConfig(gas_enabled=True, porosity=0.2, yield_stress=100,
        viscosity=1e8, gas_production_rate=0))
    for index in range(30):
        stretch = 1 - 0.001 * (index + 1)
        state = PointStep(material, state, np.eye(3) * stretch, (index + 1) * 0.001, 0.001)
        internal = state.material_state
        assert 0 < internal.porosities.numpy()[0] <= 0.200001
        np.testing.assert_allclose(internal.porosities.numpy(), np.linalg.det(internal.plastic_gradients.numpy()) - 1 + 0.2, atol=2e-6)
        assert internal.porosity_residuals.numpy()[0] < 1e-6
    assert internal.porosities.numpy()[0] < 0.199
    # Resolving tiny pores is allowed without capping their physical pressure.
    material, state = MakeGasPoint(DingConfig(gas_enabled=True, porosity=1e-7, gas_production_rate=0))
    stresses = wp.zeros(1, dtype=wp.mat33, device='cpu')
    material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    assert np.isfinite(stresses.numpy()).all()
    # Fixed nonzero gas content in a much smaller pore volume remains finite,
    # but its pressure stiffness cannot be advanced with the default explicit dt.
    state.material_state.gas_moles.assign(np.array([101325 * 0.2 / (gas_constant * 293.15)], dtype=np.float64))
    material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    assert np.isfinite(stresses.numpy()).all()
    assert abs(stresses.numpy()[0, 0, 0]) > 1e10
    material.simulation_config = SimulationConfig()
    with pytest.raises(RuntimeError, match='Gas acoustic CFL'):
        material.ComputeStress(state, stresses, wp.zeros(2, dtype=int, device='cpu'))
    with pytest.raises(ValueError, match='positive initial porosity'):
        DingConfig(gas_enabled=True, porosity=0).Validate()


def BuildGasSimulator(rate=0.005, dt=0.0005, history=False, device_name='cpu'):
    mesh = CreateDemoMesh(1)
    config = DingConfig(gas_enabled=True, porosity=0.2, yield_stress=20., viscosity=1e5,
        gas_initial_pressure_pa=1000., gas_ambient_pressure_pa=1000., gas_production_rate=rate)
    return MpmSimulator(mesh, SampleParticles(mesh, 1000., 4),
        replace(SimulationConfig(), device_name=device_name, time_step=dt, gravity=(0., 0., 0.)),
        MaterialConfig(model='ding2019', young_modulus=10000.), retain_history=history, ding_config=config,
        transport_config=TransportConfig(enabled=True, water_enabled=False, initial_temperature_k=333.15,
                                        conductivity=0., heat_transfer=0., tray_heat_transfer=0.))


@wp.kernel
def VolumeLoss(gradients: wp.array(dtype=wp.mat33), loss: wp.array(dtype=float)):
    index = wp.tid()
    wp.atomic_add(loss, 0, wp.determinant(gradients[index]) / float(gradients.shape[0]))


def EvaluateGasGradient(rate, backward=False):
    simulator = BuildGasSimulator(rate=rate, history=True)
    loss = wp.zeros(1, dtype=float, device='cpu', requires_grad=True)
    tape = wp.Tape()
    with tape:
        for index in range(20):
            simulator.Step()
        wp.launch(VolumeLoss, simulator.particle_data.particle_count,
            inputs=[simulator.current_state.deformation_gradients, loss], device='cpu')
    gradient = None
    if backward:
        tape.backward(loss)
        gradient = float(simulator.material.production_parameter.grad.numpy()[0])
    return float(loss.numpy()[0]), gradient


def TestGasProductionTrajectoryGradient():
    value, gradient = EvaluateGasGradient(0.005, True)
    upper, _ = EvaluateGasGradient(0.0051)
    lower, _ = EvaluateGasGradient(0.0049)
    difference = (upper - lower) / 0.0002
    assert abs(difference) > 1e-4
    print('gas alpha trajectory:', value, 'AD:', gradient, 'FD:', difference)
    np.testing.assert_allclose(gradient, difference, rtol=0.05, atol=1e-4)


@pytest.mark.parametrize('device_name', ['cpu', 'cuda:0'])
def TestGasExpansionHistoryAndSourceEffect(tmp_path, device_name):
    if device_name.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    volumes = []
    for rate in (0.0, 0.005):
        simulator = BuildGasSimulator(rate=rate, history=True, device_name=device_name)
        simulator.Advance(100)
        volumes.append(float(np.linalg.det(simulator.current_state.deformation_gradients.numpy()).mean()))
        assert len(simulator.state_history) == 101
        np.testing.assert_allclose(simulator.state_history[0].material_state.gas_moles.numpy(),
            1000 * 0.2 / (gas_constant * 333.15))
        assert simulator.state_history[0].material_state.gas_moles.ptr != simulator.current_state.material_state.gas_moles.ptr
    assert volumes[1] > volumes[0] + 0.001
    path = tmp_path / 'gas.npz'
    simulator.SaveSnapshot(path)
    with np.load(path) as snapshot:
        for name in ('gas_moles', 'gas_pressures', 'porosity_residuals'):
            np.testing.assert_array_equal(snapshot['ding_' + name], getattr(simulator.current_state.material_state, name).numpy())
    print(device_name, 'zero-source / source volume ratios:', volumes)
    simulator.Reset()
    np.testing.assert_allclose(simulator.current_state.material_state.gas_moles.numpy(), 1000 * 0.2 / (gas_constant * 333.15))


def TestGasTimeStepSensitivityAndDeferredRecording(tmp_path):
    results = []
    for dt in (0.0005, 0.00025):
        simulator = BuildGasSimulator(dt=dt)
        simulator.Advance(round(0.05 / dt))
        results.append(float(np.linalg.det(simulator.current_state.deformation_gradients.numpy()).mean()))
    np.testing.assert_allclose(results[0], results[1], rtol=0.003)
    print('gas dt volume ratios:', results)
    config = replace(ExperimentConfig(), material=simulator.material.material_config, ding=simulator.material.ding_config)
    config = replace(config, run=replace(config.run, log_path=str(tmp_path / 'results')))
    recorder = SimulationRecorder(simulator, config)
    simulator.result_recorder = recorder
    simulator.Advance(5)
    assert not (tmp_path / 'results').exists()
    path = recorder.Finish()
    summary = json.loads((path / 'summary.json').read_text())
    assert summary['final']['gas_total_mol'] > 0
    with (path / 'material_history.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 6


@wp.kernel
def NearClosureReturn(result: wp.array(dtype=float)):
    from_value = float(3.725290298461914e-7)
    initial_porosity = float(0.2)
    volume_value = wp.float64(1.0) - wp.float64(initial_porosity) + wp.float64(from_value)
    result[0] = solve_porosity(4.115, 29.464, 0.001052, 0.003390, -0.0230067, volume_value, initial_porosity)
    result[1] = float(porosity_prediction(wp.float64(result[0]), 4.115, 29.464, 0.001052, 0.003390, -0.0230067, volume_value, initial_porosity))


def TestNearClosureConsistentReturnDoesNotOvershoot():
    result = wp.zeros(2, dtype=float, device='cpu')
    wp.launch(NearClosureReturn, 1, inputs=[result], device='cpu')
    values = result.numpy()
    assert 0 < values[0] < 3.725290298461914e-7
    np.testing.assert_allclose(values[0], values[1], atol=1e-12)
    print('near-closure consistent porosity:', values.tolist())
