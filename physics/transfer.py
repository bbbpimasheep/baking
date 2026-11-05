import warp as wp


@wp.func
def quadratic_weight(local_coord: float, stencil_index: int):
    weight_value = 0.0
    if stencil_index == 0:
        weight_value = 0.5 * (1.5 - local_coord) * (1.5 - local_coord)
    elif stencil_index == 1:
        weight_value = 0.75 - (local_coord - 1.0) * (local_coord - 1.0)
    else:
        weight_value = 0.5 * (local_coord - 0.5) * (local_coord - 0.5)
    return weight_value


@wp.func
def stencil_base(grid_position: wp.vec3):
    return wp.vec3i(int(wp.floor(grid_position[0] - 0.5)),
                    int(wp.floor(grid_position[1] - 0.5)),
                    int(wp.floor(grid_position[2] - 0.5)))


@wp.func
def valid_stencil(base_coord: wp.vec3i, grid_resolution: int):
    return (base_coord[0] >= 0 and base_coord[1] >= 0 and base_coord[2] >= 0
            and base_coord[0] + 2 < grid_resolution
            and base_coord[1] + 2 < grid_resolution
            and base_coord[2] + 2 < grid_resolution)


@wp.func
def flatten_node(node_coord: wp.vec3i, grid_resolution: int):
    return (node_coord[0] * grid_resolution + node_coord[1]) * grid_resolution + node_coord[2]


@wp.kernel
def ParticleToGrid(particle_positions: wp.array(dtype=wp.vec3),
                   particle_velocities: wp.array(dtype=wp.vec3),
                   affine_velocities: wp.array(dtype=wp.mat33),
                   kirchhoff_stresses: wp.array(dtype=wp.mat33),
                   particle_masses: wp.array(dtype=float),
                   reference_volumes: wp.array(dtype=float),
                   node_masses: wp.array(dtype=float),
                   node_momenta: wp.array(dtype=wp.vec3),
                   grid_origin: wp.vec3, grid_spacing: float, grid_resolution: int,
                   time_step: float, error_flags: wp.array(dtype=int)):
    particle_index = wp.tid()
    grid_position = (particle_positions[particle_index] - grid_origin) / grid_spacing
    base_coord = stencil_base(grid_position)
    if not valid_stencil(base_coord, grid_resolution):
        wp.atomic_max(error_flags, 1, 1)
        return
    local_position = grid_position - wp.vec3(base_coord)
    particle_mass = particle_masses[particle_index]
    affine_matrix = (particle_mass * affine_velocities[particle_index]
                     - time_step * reference_volumes[particle_index] * 4.0 /
                     (grid_spacing * grid_spacing) * kirchhoff_stresses[particle_index])
    for x_offset in range(3):
        for y_offset in range(3):
            for z_offset in range(3):
                stencil_offset = wp.vec3i(x_offset, y_offset, z_offset)
                weight_value = (quadratic_weight(local_position[0], x_offset)
                                * quadratic_weight(local_position[1], y_offset)
                                * quadratic_weight(local_position[2], z_offset))
                node_index = flatten_node(base_coord + stencil_offset, grid_resolution)
                node_displacement = (wp.vec3(stencil_offset) - local_position) * grid_spacing
                node_momentum = weight_value * (particle_mass * particle_velocities[particle_index]
                                                + affine_matrix * node_displacement)
                wp.atomic_add(node_masses, node_index, weight_value * particle_mass)
                wp.atomic_add(node_momenta, node_index, node_momentum)


@wp.kernel
def GridToParticle(input_positions: wp.array(dtype=wp.vec3),
                   input_gradients: wp.array(dtype=wp.mat33),
                   node_velocities: wp.array(dtype=wp.vec3),
                   output_positions: wp.array(dtype=wp.vec3),
                   output_velocities: wp.array(dtype=wp.vec3),
                   output_affines: wp.array(dtype=wp.mat33),
                   output_gradients: wp.array(dtype=wp.mat33),
                   grid_origin: wp.vec3, grid_spacing: float, grid_resolution: int,
                   time_step: float, error_flags: wp.array(dtype=int)):
    particle_index = wp.tid()
    grid_position = (input_positions[particle_index] - grid_origin) / grid_spacing
    base_coord = stencil_base(grid_position)
    if not valid_stencil(base_coord, grid_resolution):
        wp.atomic_max(error_flags, 1, 1)
        return
    local_position = grid_position - wp.vec3(base_coord)
    new_velocity = wp.vec3(0.0)
    new_affine = wp.mat33(0.0)
    for x_offset in range(3):
        for y_offset in range(3):
            for z_offset in range(3):
                stencil_offset = wp.vec3i(x_offset, y_offset, z_offset)
                weight_value = (quadratic_weight(local_position[0], x_offset)
                                * quadratic_weight(local_position[1], y_offset)
                                * quadratic_weight(local_position[2], z_offset))
                node_index = flatten_node(base_coord + stencil_offset, grid_resolution)
                node_displacement = (wp.vec3(stencil_offset) - local_position) * grid_spacing
                node_velocity = node_velocities[node_index]
                new_velocity = new_velocity + weight_value * node_velocity
                new_affine = new_affine + (4.0 / (grid_spacing * grid_spacing) * weight_value
                                           * wp.outer(node_velocity, node_displacement))
    output_velocities[particle_index] = new_velocity
    output_affines[particle_index] = new_affine
    output_positions[particle_index] = input_positions[particle_index] + time_step * new_velocity
    output_gradients[particle_index] = ((wp.identity(n=3, dtype=float) + time_step * new_affine)
                                       * input_gradients[particle_index])
