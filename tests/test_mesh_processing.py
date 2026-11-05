import numpy as np
import pytest
from geometry.mesh_loader import CreateDemoMesh, LoadTetMesh
from geometry.mesh_validation import ComputeSignedVolumes, ValidateTetMesh
from geometry.particle_sampling import SampleParticles


def TestDemoVolumeAndSampling():
    tet_mesh = CreateDemoMesh(2)
    np.testing.assert_allclose(tet_mesh.reference_volumes.sum(), 0.3 ** 3)
    assert np.all(ComputeSignedVolumes(tet_mesh.vertex_positions, tet_mesh.tet_indices) > 0.0)
    particle_samples = SampleParticles(tet_mesh, 1000.0, particles_per_tet=4)
    np.testing.assert_allclose(particle_samples.reference_volumes.sum(), 0.3 ** 3)
    np.testing.assert_allclose(particle_samples.particle_masses.sum(), 27.0)
    for tet_index in range(len(tet_mesh.tet_indices)):
        tet_positions = tet_mesh.vertex_positions[tet_mesh.tet_indices[tet_index]]
        particle_positions = particle_samples.particle_positions[particle_samples.source_tets == tet_index]
        barycentric_coords = np.linalg.solve((tet_positions[1:] - tet_positions[0]).T,
                                             (particle_positions - tet_positions[0]).T).T
        assert np.all(barycentric_coords >= 0.0)
        assert np.all(barycentric_coords.sum(axis=1) <= 1.0)


def TestInvalidMeshRejected():
    tet_mesh = CreateDemoMesh(1)
    with pytest.raises(ValueError, match="repeated"):
        ValidateTetMesh(tet_mesh.vertex_positions, np.array([[0, 0, 1, 2]]))
    with pytest.raises(ValueError, match="degenerate"):
        ValidateTetMesh(np.zeros((4, 3)), np.array([[0, 1, 2, 3]]))


def TestMeshFormatsRoundTrip(tmp_path):
    import meshio
    tet_mesh = CreateDemoMesh(1)
    npz_path = tmp_path / "input.npz"
    np.savez(npz_path, vertices=tet_mesh.vertex_positions, tetrahedra=tet_mesh.tet_indices)
    loaded_mesh = LoadTetMesh(npz_path)
    np.testing.assert_allclose(loaded_mesh.reference_volumes, tet_mesh.reference_volumes)
    for file_suffix in (".mesh", ".vtu", ".msh"):
        mesh_path = tmp_path / ("input" + file_suffix)
        meshio.write_points_cells(mesh_path, tet_mesh.vertex_positions,
                                  [("tetra", tet_mesh.tet_indices)])
        loaded_mesh = LoadTetMesh(mesh_path)
        np.testing.assert_allclose(loaded_mesh.vertex_positions, tet_mesh.vertex_positions)
        np.testing.assert_allclose(loaded_mesh.reference_volumes, tet_mesh.reference_volumes)


def TestAutomaticSamplingAndLimit():
    tet_mesh = CreateDemoMesh(1)
    particle_samples = SampleParticles(tet_mesh, 1000.0, target_volume=0.001)
    assert len(particle_samples.particle_positions) >= 24
    np.testing.assert_allclose(particle_samples.reference_volumes.sum(), 0.027)
    with pytest.raises(ValueError, match="max_particles"):
        SampleParticles(tet_mesh, 1000.0, target_volume=1.0e-12)
