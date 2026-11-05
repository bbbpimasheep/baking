from dataclasses import replace, fields
import numpy as np
import warp as wp
from config import TransportConfig, DingConfig, MaterialConfig, SimulationConfig, ExperimentConfig, RunConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator
from physics.thermal_water import BoundaryFaces, SurfaceAreas, WaterStress, DiffuseGrid
from experiments.recording import SimulationRecorder


def BuildTransport(config=None, history=False, spacing=0.025, dt=0.0001, subdivisions=2, gas=False):
    mesh = CreateDemoMesh(subdivisions)
    return MpmSimulator(mesh, SampleParticles(mesh, 1000., 4),
        replace(SimulationConfig(), device_name='cpu', time_step=dt, gravity=(0., 0., 0.),
                grid_spacing=spacing, grid_resolution=int(round(1 / spacing))),
        MaterialConfig(model='ding2019', young_modulus=10000.), retain_history=history,
        ding_config=DingConfig(gas_enabled=gas, porosity=0.2 if gas else 0., gas_production_rate=0.001,
                              temperature_enabled=True),
        transport_config=config or TransportConfig(enabled=True))


def ThermalSteps(simulator, count, dt):
    # Freeze geometry to isolate transport and conservation from mechanical work.
    for _ in range(count):
        simulator.current_state = simulator.transport.Advance(simulator.current_state, dt)
    return simulator.current_state.transport_state


def TestInsulatedUniformTemperatureAndWaterConservation():
    config = TransportConfig(enabled=True, heat_transfer=0., tray_heat_transfer=0., surface_water_loss=0.)
    simulator = BuildTransport(config)
    initial = simulator.current_state.transport_state
    initial_water = initial.water_masses.numpy().sum()
    ThermalSteps(simulator, 8, 0.001)
    result = simulator.current_state.transport_state
    np.testing.assert_allclose(result.temperatures.numpy(), config.initial_temperature_k, atol=1e-12)
    np.testing.assert_allclose(result.water_masses.numpy().sum(), initial_water, rtol=2e-7)
    # Nonuniform water redistributes internally without changing total mass.
    masses = result.water_masses.numpy()
    positions = simulator.current_state.particle_positions.numpy()
    masses *= np.where(positions[:, 0] < 0.5, 1.5, 0.5)
    result.water_masses.assign(masses)
    before = masses.sum()
    after = ThermalSteps(simulator, 8, 0.01).water_masses.numpy()
    np.testing.assert_allclose(after.sum(), before, rtol=2e-7)
    assert (after[masses > masses.mean()] - masses[masses > masses.mean()]).sum() < 0


def TestBoundaryAreaTrayAndWaterStress():
    simulator = BuildTransport()
    faces, owners = BoundaryFaces(simulator.tet_mesh, simulator.particle_samples)
    assert len(faces) == 48
    count = simulator.particle_data.particle_count
    exposed = wp.zeros(count, dtype=wp.float64, device='cpu')
    contact = wp.zeros(count, dtype=wp.float64, device='cpu')
    wp.launch(SurfaceAreas, len(faces), inputs=[simulator.vertex_positions, simulator.transport.faces,
        simulator.transport.owners, exposed, contact, .45, .001], device='cpu')
    np.testing.assert_allclose(exposed.numpy().sum(), .45, rtol=2e-6)
    np.testing.assert_allclose(contact.numpy().sum(), .09, rtol=2e-6)
    before = wp.zeros(count, dtype=wp.mat33, device='cpu')
    after = wp.zeros_like(before)
    wp.launch(WaterStress, count, inputs=[simulator.current_state.deformation_gradients,
        wp.full(count, .3, dtype=wp.float64, device='cpu'), before, after, .4, 10000.], device='cpu')
    np.testing.assert_allclose(after.numpy(), np.tile(np.eye(3) * 1000, (count,1,1)), rtol=1e-6)


def TestOutsideInWaterLossHeatBalanceAndLatentCooling():
    common = TransportConfig(enabled=True, conductivity=20., specific_heat=100., heat_transfer=2000.,
                             surface_water_loss=1e-5, water_diffusivity=1e-8, water_stress=0.)
    temperatures = []
    for latent in (0., 2.26e6):
        simulator = BuildTransport(replace(common, latent_heat=latent), subdivisions=3)
        state = ThermalSteps(simulator, 40, .001)
        t = state.temperatures.numpy()
        p = simulator.particle_samples.particle_positions
        surface = np.zeros(len(t), bool)
        surface[simulator.transport.owners.numpy()] = True
        core = np.linalg.norm(p - np.array([.5,.6,.5]), axis=1) < .07
        assert t[surface].mean() > t[core].mean()
        assert state.water_masses.numpy().sum() < common.water_density * common.initial_saturation * .027
        np.testing.assert_allclose(state.water_masses.numpy().sum() + state.water_losses.numpy().sum(),
                                   common.water_density * common.initial_saturation * .027, rtol=2e-7)
        np.testing.assert_allclose(np.dot(simulator.transport.capacities.numpy(), t - common.initial_temperature_k),
                                   state.heat_exchanges.numpy().sum(), rtol=2e-6)
        temperatures.append(t.mean())
    assert temperatures[1] < temperatures[0]


@wp.kernel
def MeanTemperature(temperatures: wp.array(dtype=wp.float64), loss: wp.array(dtype=wp.float64)):
    index = wp.tid()
    wp.atomic_add(loss, 0, temperatures[index] / wp.float64(temperatures.shape[0]))


def TemperatureGradient(transfer, backward=False):
    config = TransportConfig(enabled=True, heat_transfer=transfer, water_stress=0.)
    simulator = BuildTransport(config, history=True, gas=True)
    initial = simulator.current_state
    old_temperature = initial.transport_state.temperatures.numpy().copy()
    loss = wp.zeros(1, dtype=wp.float64, device='cpu', requires_grad=True)
    with wp.Tape() as tape:
        for _ in range(4):
            simulator.Step()
        wp.launch(MeanTemperature, simulator.particle_data.particle_count,
                  inputs=[simulator.current_state.transport_state.temperatures, loss], device='cpu')
    if backward:
        tape.backward(loss)
    np.testing.assert_array_equal(initial.transport_state.temperatures.numpy(), old_temperature)
    assert len(simulator.transport_workspace_history) == 4
    for step in simulator.state_history:
        assert step.transport_state is not None
        assert all(getattr(step.transport_state, f.name) is not None for f in fields(step.transport_state))
    return loss.numpy()[0], simulator.transport.heat_transfer_parameter.grad.numpy()[0] if backward else None


def TestLocalTemperatureGasHistoryAndGradient():
    value, automatic = TemperatureGradient(20., True)
    finite = (TemperatureGradient(22.)[0] - TemperatureGradient(18.)[0]) / 4.
    assert automatic > 0
    np.testing.assert_allclose(automatic, finite, rtol=3e-3, atol=1e-10)


def TestLocalTemperatureChangesGasAndPlasticParameters():
    from physics.materials.ding import TemperatureFactor
    from physics.materials.gas import gas_constant
    config = TransportConfig(enabled=True, conductivity=0., heat_transfer=0., tray_heat_transfer=0.,
                             water_diffusivity=0., surface_water_loss=0., water_stress=0.)
    simulator = BuildTransport(config, gas=True)
    state = simulator.current_state
    count = simulator.particle_data.particle_count
    values = np.linspace(300., 360., count)
    state.transport_state.temperatures.assign(values)
    before = state.material_state.gas_moles.numpy().copy()
    dt = simulator.simulation_config.time_step
    simulator.Step()
    internal = simulator.current_state.material_state
    ding = simulator.material.ding_config
    expected = before + dt * 1000 * ding.gas_production_rate * np.exp(-((values-ding.gas_reaction_temperature_k)/ding.gas_reaction_width_k)**2)
    np.testing.assert_allclose(internal.gas_moles.numpy() - before, expected - before, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(internal.temperatures.numpy(), values, atol=2e-5)
    expected_yield = [ding.yield_stress * TemperatureFactor(t, ding.temperature_knots_k, ding.yield_factors) for t in values]
    np.testing.assert_allclose(internal.yield_stresses.numpy(), expected_yield, rtol=1e-5)
    np.testing.assert_array_equal(internal.gelled.numpy(), values >= ding.temperature_knots_k[-1])
    gradients = simulator.current_state.deformation_gradients.numpy()
    expected_pressure = expected * gas_constant * internal.temperatures.numpy() / (np.linalg.det(gradients) * internal.porosities.numpy())
    np.testing.assert_allclose(internal.gas_pressures.numpy(), expected_pressure, rtol=5e-7)


def TestWaterDisabledAndDeferredRecording(tmp_path):
    config = TransportConfig(enabled=True, water_enabled=False)
    simulator = BuildTransport(config)
    reference = BuildTransport(config)
    experiment = ExperimentConfig(material=simulator.material.material_config, simulation=simulator.simulation_config,
        ding=simulator.material.ding_config, transport=config,
        run=RunConfig(log_path=str(tmp_path/'results'), snapshot_interval=2))
    recorder = SimulationRecorder(simulator, experiment)
    simulator.result_recorder = recorder
    initial = simulator.current_state.transport_state.water_masses.numpy().copy()
    for _ in range(3):
        simulator.Step()
        reference.Step()
    np.testing.assert_array_equal(simulator.current_state.particle_positions.numpy(), reference.current_state.particle_positions.numpy())
    np.testing.assert_array_equal(simulator.current_state.transport_state.water_masses.numpy(), initial)
    assert not (tmp_path/'results').exists()
    path = recorder.Finish()
    assert (path/'thermal_water_history.csv').exists()
    assert (path/'snapshot_0000000.npz').exists()
    assert (path/'snapshot_0000002.npz').exists()
    with np.load(path/'final_state.npz') as snapshot:
        assert 'transport_water_masses' in snapshot and 'transport_temperatures' in snapshot


def TestExplicitTransportBound():
    simulator = BuildTransport(TransportConfig(enabled=True, conductivity=1e8))
    import pytest
    with pytest.raises(RuntimeError, match='stability bound'):
        simulator.transport.Advance(simulator.current_state, .1)


def GridDecay(resolution, dt):
    # Exact insulated cosine heat mode: T=300+10*cos(pi*x)*exp(-D*pi^2*t).
    count = resolution ** 3
    spacing = 1. / resolution
    coords = np.indices((resolution,) * 3)[0].reshape(-1)
    initial = 300 + 10 * np.cos(np.pi * (coords + .5) * spacing)
    temperature = wp.array(initial, dtype=wp.float64, device='cpu')
    volume = wp.full(count, spacing**3, dtype=wp.float64, device='cpu')
    capacities = wp.full(count, 1000 * spacing**3, dtype=wp.float64, device='cpu')
    zero = wp.zeros(count, dtype=wp.float64, device='cpu')
    changes = wp.zeros_like(zero)
    water_changes = wp.zeros_like(zero)
    flags = wp.zeros(2, dtype=int, device='cpu')
    for _ in range(round(.2 / dt)):
        wp.launch(DiffuseGrid, count, inputs=[volume, capacities, temperature, zero, zero, zero, zero, zero,
            changes, water_changes, flags, 10., 0., 1000., 0, spacing, resolution, dt], device='cpu')
        # Frozen scalar verification does not require a differentiation tape.
        temperature.assign(temperature.numpy() + changes.numpy())
    exact = 300 + (initial - 300) * np.exp(-.01 * np.pi**2 * .2)
    assert flags.numpy().max() == 0
    amplitude = np.dot(temperature.numpy() - 300, initial - 300) / np.dot(initial - 300, initial - 300)
    return float(np.sqrt(np.mean((temperature.numpy() - exact)**2))), float(amplitude)


def TestGridAndTimeConvergence():
    errors = [GridDecay(n, .0005)[0] for n in (8, 16, 32)]
    assert errors[1] < .35 * errors[0] and errors[2] < .35 * errors[1]
    # Compare to the discrete eigenmode to isolate first-order time error.
    n = 16
    rate = 4 * .01 * n*n * np.sin(np.pi/(2*n))**2
    amplitude_exact = np.exp(-rate * .2)
    time_errors = [abs(GridDecay(n, dt)[1] - amplitude_exact) for dt in (.02,.01,.005)]
    assert time_errors[1] < .51 * time_errors[0] and time_errors[2] < .51 * time_errors[1]
