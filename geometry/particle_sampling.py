from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ParticleSamples:
    particle_positions: np.ndarray
    reference_volumes: np.ndarray
    particle_masses: np.ndarray
    source_tets: np.ndarray


def SampleParticles(tet_mesh, density, particles_per_tet=0, target_volume=None,
                    random_seed=42, max_particles=500000):
    """Four symmetric samples, or deterministic seeded uniform volume samples.

    particles_per_tet=0 chooses >=4 samples/tet from target_volume. Mesh topology
    and sampling remain fixed during a simulation and future differentiation.
    """
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("density must be positive and finite")
    if particles_per_tet < 0:
        raise ValueError("particles_per_tet must be nonnegative")
    if particles_per_tet == 0:
        if target_volume is None or not np.isfinite(target_volume) or target_volume <= 0.0:
            raise ValueError("automatic sampling needs a positive target_volume")
        raw_counts = np.maximum(4.0, np.ceil(tet_mesh.reference_volumes / target_volume))
        if raw_counts.sum() > max_particles:
            raise ValueError("sampling exceeds max_particles; coarsen the input mesh or sampling")
        sample_counts = raw_counts.astype(np.int64)
    else:
        sample_counts = np.full(len(tet_mesh.tet_indices), particles_per_tet, dtype=np.int64)
    if sample_counts.sum() > max_particles:
        raise ValueError("sampling exceeds max_particles")
    source_tets = np.repeat(np.arange(len(sample_counts)), sample_counts)
    particle_positions = np.empty((len(source_tets), 3), dtype=np.float64)
    reference_volumes = np.repeat(tet_mesh.reference_volumes / sample_counts, sample_counts)
    random_generator = np.random.default_rng(random_seed)
    symmetric_weights = np.full((4, 4), (5.0 - np.sqrt(5.0)) / 20.0)
    np.fill_diagonal(symmetric_weights, (5.0 + 3.0 * np.sqrt(5.0)) / 20.0)
    particle_offset = 0
    for tet_index, sample_count in enumerate(sample_counts):
        if sample_count == 4:
            barycentric_weights = symmetric_weights
        elif sample_count == 1:
            barycentric_weights = np.full((1, 4), 0.25)
        else:
            barycentric_weights = random_generator.dirichlet(np.ones(4), size=sample_count)
        particle_positions[particle_offset:particle_offset + sample_count] = (
            barycentric_weights @ tet_mesh.vertex_positions[tet_mesh.tet_indices[tet_index]])
        particle_offset += sample_count
    return ParticleSamples(particle_positions, reference_volumes,
                           density * reference_volumes, source_tets)
