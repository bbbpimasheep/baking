import warp as wp


class GridState:
    def __init__(self, simulation_config, device_name, requires_grad=False):
        self.node_count = simulation_config.grid_resolution ** 3
        self.node_masses = wp.zeros(self.node_count, dtype=float, device=device_name,
                                   requires_grad=requires_grad)
        self.node_momenta = wp.zeros(self.node_count, dtype=wp.vec3, device=device_name,
                                    requires_grad=requires_grad)
        self.node_velocities = wp.zeros(self.node_count, dtype=wp.vec3, device=device_name,
                                       requires_grad=requires_grad)

    def Clear(self):
        self.node_masses.zero_()
        self.node_momenta.zero_()
        self.node_velocities.zero_()


@wp.kernel
def UpdateGrid(node_masses: wp.array(dtype=float),
               node_momenta: wp.array(dtype=wp.vec3),
               node_velocities: wp.array(dtype=wp.vec3),
               grid_resolution: int, boundary_cells: int,
               gravity: wp.vec3, time_step: float):
    node_index = wp.tid()
    if node_masses[node_index] > 0.0:
        node_velocity = node_momenta[node_index] / node_masses[node_index] + time_step * gravity
        node_coords = wp.vec3i(node_index // (grid_resolution * grid_resolution),
                               (node_index // grid_resolution) % grid_resolution,
                               node_index % grid_resolution)
        # Frictionless, unilateral grid-wall velocity projection.
        for axis_index in range(3):
            if node_coords[axis_index] < boundary_cells and node_velocity[axis_index] < 0.0:
                node_velocity[axis_index] = 0.0
            if node_coords[axis_index] >= grid_resolution - boundary_cells and node_velocity[axis_index] > 0.0:
                node_velocity[axis_index] = 0.0
        node_velocities[node_index] = node_velocity
