from dataclasses import replace
from pathlib import Path
import numpy as np
from config import DingConfig, LoadExperimentConfig, MaterialConfig, SimulationConfig
from experiments.recording import ExtractSurface
from geometry.mesh_loader import CreateSphereMesh, LoadTetMesh
from geometry.mesh_validation import ComputeSignedVolumes
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def TestSphereBoundaryVolumeAndNearFloorPlacement():
    mesh = CreateSphereMesh()
    vertices, faces = ExtractSurface(mesh.vertex_positions, mesh.tet_indices)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 0.1, atol=1e-12)
    np.testing.assert_allclose(mesh.vertex_positions.min(axis=0), -0.1, atol=1e-12)
    np.testing.assert_allclose(mesh.vertex_positions.max(axis=0), 0.1, atol=1e-12)
    assert np.all(ComputeSignedVolumes(mesh.vertex_positions, mesh.tet_indices) > 0)
    np.testing.assert_allclose(mesh.reference_volumes.sum(), 4 * np.pi * 0.1 ** 3 / 3, rtol=0.01)
    config = LoadExperimentConfig(Path(__file__).resolve().parents[1] / 'examples' / 'oven_baking.txt')
    loaded_mesh = LoadTetMesh(config.mesh.mesh_path, config.mesh.mesh_scale, config.mesh.mesh_translation)
    floor_height = (config.simulation.grid_origin[1] +
                    (config.simulation.boundary_cells - 1) * config.simulation.grid_spacing)
    assert 0.0 < loaded_mesh.vertex_positions[:, 1].min() - floor_height <= .001 + 1e-12
    samples = SampleParticles(loaded_mesh, config.material.density, particles_per_tet=3)
    simulator = MpmSimulator(loaded_mesh, samples, replace(config.simulation, device_name='cpu'),
                             config.material, ding_config=config.ding, transport_config=config.transport)
    simulator.Advance(3)


def TestNearFloorSphereDropsAndContacts():
    from geometry.mesh_validation import ValidateTetMesh
    mesh = CreateSphereMesh(subdivision_count=4)
    mesh = ValidateTetMesh(mesh.vertex_positions + np.array([0.5, 0.151, 0.5]), mesh.tet_indices)
    samples = SampleParticles(mesh, 1200.0, particles_per_tet=4)
    simulator = MpmSimulator(mesh, samples, replace(SimulationConfig(), device_name='cpu'),
        MaterialConfig(density=1200.0, young_modulus=60000.0, poisson_ratio=0.45, model='ding2019'),
        ding_config=DingConfig(yield_stress=7000.0))
    initial_height = float(simulator.current_state.particle_positions.numpy()[:, 1].mean())
    stats = simulator.Advance(1000)
    assert simulator.current_state.particle_positions.numpy()[:, 1].mean() < initial_height
    resolution = simulator.simulation_config.grid_resolution
    layer_count = simulator.simulation_config.boundary_cells
    masses = simulator.grid_state.node_masses.numpy().reshape((resolution,) * 3)
    velocities = simulator.grid_state.node_velocities.numpy().reshape((resolution,) * 3 + (3,))
    assert np.any(masses[:, :layer_count, :] > 0)
    assert np.all(velocities[:, :layer_count, :, 1] >= 0)
    assert stats['minimum_j'] > 0
