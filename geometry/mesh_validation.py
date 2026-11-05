from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TetMesh:
    vertex_positions: np.ndarray
    tet_indices: np.ndarray
    reference_volumes: np.ndarray


def ComputeSignedVolumes(vertex_positions, tet_indices):
    tet_positions = vertex_positions[tet_indices]
    edge_matrix = tet_positions[:, 1:] - tet_positions[:, :1]
    return np.linalg.det(edge_matrix) / 6.0


def ValidateTetMesh(vertex_positions, tet_indices):
    vertex_positions = np.asarray(vertex_positions, dtype=np.float64)
    raw_indices = np.asarray(tet_indices)
    if vertex_positions.ndim != 2 or vertex_positions.shape[1] != 3:
        raise ValueError("vertices must have shape (vertex_count, 3)")
    if raw_indices.ndim != 2 or raw_indices.shape[1] != 4 or len(raw_indices) == 0:
        raise ValueError("tetrahedra must have nonempty shape (tet_count, 4)")
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("tetrahedron indices must be integers (zero based)")
    if len(vertex_positions) < 4 or not np.isfinite(vertex_positions).all():
        raise ValueError("vertices must be finite and contain at least four entries")
    if raw_indices.min() < 0 or raw_indices.max() >= len(vertex_positions):
        raise ValueError("tetrahedron index outside vertex array")
    tet_indices = raw_indices.astype(np.int32, copy=True)
    sorted_indices = np.sort(tet_indices, axis=1)
    if np.any(np.diff(sorted_indices, axis=1) == 0):
        raise ValueError("tetrahedron contains repeated vertex indices")
    if len(np.unique(sorted_indices, axis=0)) != len(tet_indices):
        raise ValueError("duplicate tetrahedra detected")
    signed_volumes = ComputeSignedVolumes(vertex_positions, tet_indices)
    tet_positions = vertex_positions[tet_indices]
    edge_scale = np.max(np.linalg.norm(tet_positions[:, :, None] -
                                      tet_positions[:, None, :], axis=-1), axis=(1, 2))
    if np.any(np.abs(signed_volumes) <= 1.0e-12 * edge_scale ** 3):
        raise ValueError("zero-volume or nearly degenerate tetrahedron detected")
    # Normalize orientation without moving vertices or changing geometry.
    negative_mask = signed_volumes < 0.0
    tet_indices[negative_mask, 0], tet_indices[negative_mask, 1] = (
        tet_indices[negative_mask, 1].copy(), tet_indices[negative_mask, 0].copy())
    return TetMesh(vertex_positions.copy(), tet_indices, np.abs(signed_volumes))
