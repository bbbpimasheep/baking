from dataclasses import replace
import numpy as np
import pytest
import warp as wp

from config import MaterialConfig, SimulationConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.mesh_tracking import ParticleMeshBinding
from geometry.mesh_validation import ValidateTetMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator
from physics.state import CreateParticleState


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
@pytest.mark.parametrize("sample_count", [1, 4, 7])
def TestReconstructionReproducesAffineDeformation(device_name, sample_count):
    wp.init()
    if device_name.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    tet_mesh = CreateDemoMesh(2)
    particle_samples = SampleParticles(tet_mesh, 1000.0, sample_count)
    particle_state, _ = CreateParticleState(particle_samples, device_name)
    mesh_binding = ParticleMeshBinding(tet_mesh, particle_samples, device_name)
    output_vertices = wp.zeros(len(tet_mesh.vertex_positions), dtype=wp.vec3, device=device_name)
    rotation_angle = 0.7
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0.0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0.0],
                                [0.0, 0.0, 1.0]])
    affine_matrices = [np.eye(3), rotation_matrix,
                       rotation_matrix @ np.array([[1.2, 0.2, 0.0], [0.0, 0.8, 0.1], [0.0, 0.0, 1.1]])]
    translation_vector = np.array([0.15, -0.1, 0.05])
    for affine_matrix in affine_matrices:
        particle_state.particle_positions.assign(
            (particle_samples.particle_positions @ affine_matrix.T + translation_vector).astype(np.float32))
        particle_state.deformation_gradients.assign(
            np.tile(affine_matrix.astype(np.float32), (len(particle_samples.particle_positions), 1, 1)))
        mesh_binding.Reconstruct(particle_state, output_vertices)
        np.testing.assert_allclose(output_vertices.numpy(),
                                   tet_mesh.vertex_positions @ affine_matrix.T + translation_vector,
                                   rtol=1.0e-6, atol=4.0e-7)


def TestSharedVerticesBlendIncidentParticlePredictions():
    wp.init()
    tet_mesh = ValidateTetMesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                         [0, 0, 1], [0, 0, -2], [5, 5, 5]], dtype=float),
                               np.array([[0, 1, 2, 3], [0, 2, 1, 4]]))
    particle_samples = SampleParticles(tet_mesh, 1000.0, 1)
    particle_state, _ = CreateParticleState(particle_samples, "cpu")
    mesh_binding = ParticleMeshBinding(tet_mesh, particle_samples, "cpu")
    particle_positions = particle_samples.particle_positions.copy()
    particle_positions[0] += [0.1, 0.0, 0.0]
    particle_positions[1] += [0.0, 0.2, 0.0]
    particle_state.particle_positions.assign(particle_positions.astype(np.float32))
    deformation_gradients = np.array([np.diag([1.1, 0.9, 1.0]), np.diag([0.8, 1.2, 1.0])])
    particle_state.deformation_gradients.assign(deformation_gradients.astype(np.float32))
    output_vertices = wp.zeros(len(tet_mesh.vertex_positions), dtype=wp.vec3, device="cpu")
    mesh_binding.Reconstruct(particle_state, output_vertices)
    expected_vertices = tet_mesh.vertex_positions.copy()
    for vertex_index in range(len(expected_vertices)):
        incident_particles = [particle_index for particle_index, tet_index in
                              enumerate(particle_samples.source_tets)
                              if vertex_index in tet_mesh.tet_indices[tet_index]]
        if incident_particles:
            predictions = [particle_positions[particle_index] + deformation_gradients[particle_index] @
                           (tet_mesh.vertex_positions[vertex_index] -
                            particle_samples.particle_positions[particle_index])
                           for particle_index in incident_particles]
            expected_vertices[vertex_index] = np.average(predictions, axis=0,
                weights=particle_samples.reference_volumes[incident_particles])
    np.testing.assert_allclose(output_vertices.numpy(), expected_vertices, atol=3.0e-7)
    assert mesh_binding.unmapped_vertex_count == 1
    np.testing.assert_array_equal(output_vertices.numpy()[5], tet_mesh.vertex_positions[5])


def TestMeshUpdatesAfterSubstepsAndSnapshot(tmp_path):
    tet_mesh = CreateDemoMesh(2)
    particle_samples = SampleParticles(tet_mesh, 1000.0, 4)
    initial_velocity = np.array([0.2, 0.1, -0.05])
    simulator = MpmSimulator(tet_mesh, particle_samples,
                             replace(SimulationConfig(), gravity=(0.0, 0.0, 0.0), device_name="cpu"),
                             MaterialConfig(), initial_velocity)
    for frame_index in range(3):
        simulator.Advance(5)
        np.testing.assert_allclose(simulator.vertex_positions.numpy(),
            tet_mesh.vertex_positions + simulator.step_count * simulator.simulation_config.time_step * initial_velocity,
            atol=2.0e-6)
        assert simulator.mesh_step_count == simulator.step_count
    # Saving after a direct Step must also reconstruct the latest state.
    simulator.Step()
    output_path = tmp_path / "mesh_snapshot.npz"
    simulator.SaveSnapshot(output_path)
    with np.load(output_path) as snapshot_data:
        np.testing.assert_allclose(snapshot_data["vertices"],
            tet_mesh.vertex_positions + simulator.step_count * simulator.simulation_config.time_step * initial_velocity,
            atol=2.0e-6)
        np.testing.assert_array_equal(snapshot_data["tetrahedra"], tet_mesh.tet_indices)
        np.testing.assert_array_equal(snapshot_data["reference_vertices"], tet_mesh.vertex_positions)
    simulator.Reset()
    np.testing.assert_array_equal(simulator.vertex_positions.numpy(), tet_mesh.vertex_positions.astype(np.float32))
    assert simulator.mesh_step_count == 0
