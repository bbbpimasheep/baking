from dataclasses import replace
import numpy as np
import pytest
from config import MaterialConfig, SimulationConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def BuildSimulator(gravity=(0.0, 0.0, 0.0), initial_velocity=(0.2, -0.1, 0.05), time_step=0.0001):
    tet_mesh = CreateDemoMesh(3)
    simulation_config = replace(SimulationConfig(), device_name="cpu", gravity=gravity,
                                time_step=time_step)
    return MpmSimulator(tet_mesh, SampleParticles(tet_mesh, 1000.0, 4),
                         simulation_config, MaterialConfig(), initial_velocity)


def TestFreeTranslationConservesMassMomentumAndShape():
    simulator = BuildSimulator()
    initial_positions = simulator.current_state.particle_positions.numpy().copy()
    initial_vertices = simulator.vertex_positions.numpy().copy()
    initial_velocity = np.array(simulator.initial_velocity)
    initial_mass = simulator.particle_data.particle_masses.numpy().sum(dtype=np.float64)
    simulation_stats = simulator.Advance(30)
    final_positions = simulator.current_state.particle_positions.numpy()
    final_velocities = simulator.current_state.particle_velocities.numpy()
    elapsed_time = 30 * simulator.simulation_config.time_step
    np.testing.assert_allclose(final_positions, initial_positions + elapsed_time * initial_velocity,
                               atol=2.0e-6)
    np.testing.assert_allclose(final_velocities, np.tile(initial_velocity, (len(final_velocities), 1)),
                               atol=2.0e-5)
    np.testing.assert_allclose(simulator.current_state.deformation_gradients.numpy(),
                               np.tile(np.eye(3), (len(final_velocities), 1, 1)), atol=1.0e-5)
    np.testing.assert_allclose(simulator.vertex_positions.numpy(),
                               initial_vertices + elapsed_time * initial_velocity, atol=2.0e-6)
    np.testing.assert_allclose(simulation_stats["grid_mass"], initial_mass, rtol=1.0e-6)
    assert simulation_stats["unmapped_mesh_vertices"] == 0


def TestFreeFallMatchesSymplecticEuler():
    simulator = BuildSimulator(gravity=(0.0, -9.81, 0.0), initial_velocity=(0.0, 0.0, 0.0))
    initial_positions = simulator.current_state.particle_positions.numpy().copy()
    step_count = 40
    time_step = simulator.simulation_config.time_step
    simulator.Advance(step_count)
    expected_positions = initial_positions.copy()
    expected_positions[:, 1] -= 9.81 * time_step ** 2 * step_count * (step_count + 1) / 2
    expected_velocities = np.zeros_like(initial_positions)
    expected_velocities[:, 1] = -9.81 * time_step * step_count
    np.testing.assert_allclose(simulator.current_state.particle_positions.numpy(), expected_positions,
                               atol=1.5e-6)
    np.testing.assert_allclose(simulator.current_state.particle_velocities.numpy(), expected_velocities,
                               atol=1.0e-5)


def TestResetAndSnapshot(tmp_path):
    simulator = BuildSimulator()
    initial_positions = simulator.current_state.particle_positions.numpy().copy()
    simulator.Advance(5)
    output_path = tmp_path / "snapshot.npz"
    simulator.SaveSnapshot(output_path)
    with np.load(output_path) as snapshot_data:
        np.testing.assert_allclose(snapshot_data["particle_positions"],
                                   simulator.current_state.particle_positions.numpy())
        assert snapshot_data["simulation_time"] == 5 * simulator.simulation_config.time_step
    simulator.Reset()
    assert simulator.step_count == 0
    np.testing.assert_array_equal(simulator.current_state.particle_positions.numpy(), initial_positions)


def TestExcessiveTimeStepRejected():
    with pytest.raises(ValueError, match="CFL"):
        replace(SimulationConfig(), time_step=1.0).Validate(MaterialConfig())
