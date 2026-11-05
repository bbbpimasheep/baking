from pathlib import Path
import numpy as np
from geometry.mesh_validation import ValidateTetMesh


def LoadTetMesh(mesh_path, mesh_scale=1.0, mesh_translation=(0.0, 0.0, 0.0)):
    """Read meshio formats (.mesh, .msh, .vtk, .vtu), or NPZ vertices/tetrahedra.

    Coordinates are kept in their input units unless explicitly transformed.
    Only linear four-node tetrahedra are supported; surface cells are ignored.
    """
    mesh_path = Path(mesh_path)
    if not np.isfinite(mesh_scale) or mesh_scale <= 0.0:
        raise ValueError("mesh_scale must be positive and finite")
    mesh_translation = np.asarray(mesh_translation, dtype=np.float64)
    if mesh_translation.shape != (3,) or not np.isfinite(mesh_translation).all():
        raise ValueError("mesh_translation must contain three finite values")
    if mesh_path.suffix.lower() == ".npz":
        with np.load(mesh_path, allow_pickle=False) as mesh_data:
            vertex_positions = mesh_data["vertices"]
            tet_indices = mesh_data["tetrahedra"]
    else:
        import meshio
        mesh_data = meshio.read(mesh_path)
        if any(cell_block.type.startswith("tetra") and cell_block.type != "tetra"
               for cell_block in mesh_data.cells):
            raise ValueError("higher-order tetrahedra are unsupported; export linear tetrahedra")
        tet_blocks = [cell_block.data for cell_block in mesh_data.cells
                      if cell_block.type == "tetra"]
        if not tet_blocks:
            raise ValueError("input contains no linear tetrahedra")
        tet_indices = np.concatenate(tet_blocks)
        vertex_positions = mesh_data.points
    vertex_positions = np.asarray(vertex_positions, dtype=np.float64) * mesh_scale + mesh_translation
    return ValidateTetMesh(vertex_positions, tet_indices)


def CreateDemoMesh(subdivision_count=6):
    """A 0.3 m cube, initially above the lower y wall; six tets per voxel."""
    if subdivision_count < 1:
        raise ValueError("subdivision_count must be positive")
    axis_values = np.linspace(0.0, 0.3, subdivision_count + 1)
    vertex_positions = np.stack(np.meshgrid(axis_values, axis_values, axis_values,
                                          indexing="ij"), axis=-1).reshape(-1, 3)
    vertex_positions += np.array([0.35, 0.45, 0.35])
    grid_shape = (subdivision_count + 1,) * 3
    tet_patterns = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7],
                             [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])
    corner_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                               [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    tet_blocks = []
    for x_index in range(subdivision_count):
        for y_index in range(subdivision_count):
            for z_index in range(subdivision_count):
                corner_indices = np.ravel_multi_index(
                    (corner_offsets + [x_index, y_index, z_index]).T, grid_shape)
                tet_blocks.append(corner_indices[tet_patterns])
    return ValidateTetMesh(vertex_positions, np.concatenate(tet_blocks))


def CreateSphereMesh(radius=0.1, subdivision_count=12):
    """Map a structured tetrahedral cube to a sphere centered at the origin.

    Boundary vertices lie on the requested sphere; planar triangle faces
    approximate its surface. The map retains the conforming cube connectivity.
    Coordinates and radius are in metres.
    """
    cube_mesh = CreateDemoMesh(subdivision_count)
    lower_corner = cube_mesh.vertex_positions.min(axis=0)
    cube_extent = np.ptp(cube_mesh.vertex_positions, axis=0)
    unit_positions = 2.0 * (cube_mesh.vertex_positions - lower_corner) / cube_extent - 1.0
    sphere_positions = np.empty_like(unit_positions)
    for axis_index in range(3):
        other_axes = [index for index in range(3) if index != axis_index]
        first_squared = unit_positions[:, other_axes[0]] ** 2
        second_squared = unit_positions[:, other_axes[1]] ** 2
        sphere_positions[:, axis_index] = radius * unit_positions[:, axis_index] * np.sqrt(
            1.0 - first_squared / 2.0 - second_squared / 2.0 + first_squared * second_squared / 3.0)
    return ValidateTetMesh(sphere_positions, cube_mesh.tet_indices)
