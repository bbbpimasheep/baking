"""Reconstruct the original tet mesh from particle positions and total F.

Each vertex is bound to particles sampled in its incident reference tets.
A particle predicts x_vertex = x_particle + F_particle * (X_vertex-X_particle).
Reference-volume weights blend predictions into one shared vertex position.
The binding is fixed in material coordinates and never uses the MPM grid.
"""

import numpy as np
import warp as wp


@wp.kernel
def ReconstructMeshVertices(particle_positions: wp.array(dtype=wp.vec3),
                            deformation_gradients: wp.array(dtype=wp.mat33),
                            reference_vertices: wp.array(dtype=wp.vec3),
                            binding_offsets: wp.array(dtype=int),
                            binding_particles: wp.array(dtype=int),
                            binding_weights: wp.array(dtype=float),
                            reference_offsets: wp.array(dtype=wp.vec3),
                            output_vertices: wp.array(dtype=wp.vec3)):
    vertex_index = wp.tid()
    binding_start = binding_offsets[vertex_index]
    binding_end = binding_offsets[vertex_index + 1]
    if binding_start == binding_end:
        output_vertices[vertex_index] = reference_vertices[vertex_index]
        return
    anchor_particle = binding_particles[binding_start]
    anchor_position = (particle_positions[anchor_particle] +
                       deformation_gradients[anchor_particle] * reference_offsets[binding_start])
    vertex_displacement = wp.vec3(0.0)
    for binding_index in range(binding_start, binding_end):
        particle_index = binding_particles[binding_index]
        predicted_position = (particle_positions[particle_index] +
                              deformation_gradients[particle_index] * reference_offsets[binding_index])
        # Sum deviations from an anchor to reduce float32 error at high valence.
        vertex_displacement = vertex_displacement + binding_weights[binding_index] * (predicted_position - anchor_position)
    output_vertices[vertex_index] = anchor_position + vertex_displacement


class ParticleMeshBinding:
    def __init__(self, tet_mesh, particle_samples, device_name):
        vertex_count = len(tet_mesh.vertex_positions)
        particle_count = len(particle_samples.particle_positions)
        source_tets = np.asarray(particle_samples.source_tets)
        reference_volumes = np.asarray(particle_samples.reference_volumes, dtype=np.float64)
        if source_tets.shape != (particle_count,) or not np.issubdtype(source_tets.dtype, np.integer):
            raise ValueError("source_tets must contain one integer tet index per particle")
        if particle_count == 0 or source_tets.min() < 0 or source_tets.max() >= len(tet_mesh.tet_indices):
            raise ValueError("invalid particle source tetrahedra")
        if (reference_volumes.shape != (particle_count,) or
                not np.isfinite(reference_volumes).all() or np.any(reference_volumes <= 0.0)):
            raise ValueError("particle reference volumes must be positive and finite")
        tet_sample_counts = np.bincount(source_tets, minlength=len(tet_mesh.tet_indices))
        if np.any(tet_sample_counts == 0):
            raise ValueError("every tetrahedron must contain particles for mesh reconstruction")
        # CSR vertex -> incident particles. Shared vertices get a single result.
        binding_vertices = tet_mesh.tet_indices[source_tets].reshape(-1)
        binding_particles = np.repeat(np.arange(particle_count, dtype=np.int32), 4)
        binding_order = np.argsort(binding_vertices, kind="stable")
        binding_vertices = binding_vertices[binding_order]
        binding_particles = binding_particles[binding_order]
        binding_counts = np.bincount(binding_vertices, minlength=vertex_count)
        binding_offsets = np.concatenate(([0], np.cumsum(binding_counts))).astype(np.int32)
        binding_weights = reference_volumes[binding_particles]
        vertex_weights = np.bincount(binding_vertices, weights=binding_weights, minlength=vertex_count)
        binding_weights = binding_weights / vertex_weights[binding_vertices]
        reference_offsets = (tet_mesh.vertex_positions[binding_vertices] -
                             particle_samples.particle_positions[binding_particles])
        self.unmapped_vertex_count = int(np.count_nonzero(binding_counts == 0))
        self.reference_vertices = wp.array(tet_mesh.vertex_positions, dtype=wp.vec3, device=device_name)
        self.binding_offsets = wp.array(binding_offsets, dtype=int, device=device_name)
        self.binding_particles = wp.array(binding_particles, dtype=int, device=device_name)
        self.binding_weights = wp.array(binding_weights, dtype=float, device=device_name)
        self.reference_offsets = wp.array(reference_offsets, dtype=wp.vec3, device=device_name)

    def Reconstruct(self, particle_state, output_vertices):
        wp.launch(ReconstructMeshVertices, dim=len(output_vertices), inputs=[
            particle_state.particle_positions, particle_state.deformation_gradients,
            self.reference_vertices, self.binding_offsets, self.binding_particles,
            self.binding_weights, self.reference_offsets, output_vertices], device=output_vertices.device)
