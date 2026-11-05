from pathlib import Path
import numpy as np
from geometry.mesh_validation import ComputeSignedVolumes


class PolyscopeViewer:
    def __init__(self, simulator):
        import polyscope as ps
        import polyscope.imgui as psim
        self.ps = ps
        self.psim = psim
        self.simulator = simulator
        self.is_running = False
        self.last_error = ""
        self.latest_stats = None
        self.frame_steps = simulator.simulation_config.steps_per_frame

    def Initialize(self, backend_name="auto"):
        self.ps.set_program_name("Candy: 3D hyperelastic MLS-MPM")
        self.ps.init(backend=backend_name)
        self.ps.set_up_dir("y_up")
        # The visible floor marks the nominal lower grid wall layer.
        simulation_config = self.simulator.simulation_config
        floor_height = (simulation_config.grid_origin[1] +
                        (simulation_config.boundary_cells - 1) * simulation_config.grid_spacing)
        self.ps.set_ground_plane_height_mode("manual")
        self.ps.set_ground_plane_height(floor_height)
        self.particle_cloud = self.ps.register_point_cloud(
            "MPM particles", self.simulator.current_state.particle_positions.numpy(),
            radius=0.18 * simulation_config.grid_spacing, point_render_mode="sphere", enabled=False)
        self.particle_cloud.set_radius(0.18 * simulation_config.grid_spacing, relative=False)
        self.volume_mesh = self.ps.register_volume_mesh(
            "Reconstructed material mesh", self.simulator.vertex_positions.numpy(),
            tets=self.simulator.tet_mesh.tet_indices, enabled=True)
        self.RegisterDomain()
        self.Refresh()
        # Frame the material from a low, close viewpoint rather than the full grid domain.
        vertex_positions = self.simulator.vertex_positions.numpy()
        material_center = 0.5 * (vertex_positions.min(axis=0) + vertex_positions.max(axis=0))
        material_size = float(np.ptp(vertex_positions, axis=0).max())
        camera_target = material_center - np.array([0.0, 0.05 * material_size, 0.0])
        camera_position = material_center + np.array([1.2 * material_size, 0.0, 1.8 * material_size])
        camera_position[1] = floor_height + 0.6 * material_size
        self.ps.set_view_projection_mode("perspective")
        self.ps.look_at(camera_position, camera_target)
        self.ps.set_view_center_raw(camera_target)
        self.ps.set_user_callback(self.Callback)

    def RegisterDomain(self):
        simulation_config = self.simulator.simulation_config
        corner_offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                   [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        domain_vertices = (np.asarray(simulation_config.grid_origin) + corner_offsets *
                           (simulation_config.grid_resolution - 1) * simulation_config.grid_spacing)
        edge_indices = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6],
                                 [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
        self.ps.register_curve_network("Background grid domain", domain_vertices, edge_indices, enabled=False)

    def Refresh(self):
        self.simulator.UpdateMesh()
        particle_state = self.simulator.current_state
        self.particle_cloud.update_point_positions(particle_state.particle_positions.numpy())
        self.particle_cloud.add_scalar_quantity(
            "Speed (m/s)", np.linalg.norm(particle_state.particle_velocities.numpy(), axis=1))
        self.particle_cloud.add_scalar_quantity(
            "J = det(F)", np.linalg.det(particle_state.deformation_gradients.numpy()))
        if hasattr(particle_state.material_state, 'porosities'):
            material_state = particle_state.material_state
            for quantity_name, value_name in (("Porosity", "porosities"), ("Temperature (K)", "temperatures"),
                                               ("Yield stress (Pa)", "yield_stresses"), ("Gas pressure (Pa)", "gas_pressures")):
                particle_values = getattr(material_state, value_name).numpy()
                self.particle_cloud.add_scalar_quantity(quantity_name, particle_values)
                # Particle quadrature weights define a per-original-tet display average.
                weights = self.simulator.particle_samples.reference_volumes
                source_tets = self.simulator.particle_samples.source_tets
                cell_values = np.bincount(source_tets, weights=weights * particle_values,
                                         minlength=len(self.simulator.tet_mesh.tet_indices)) / self.simulator.tet_mesh.reference_volumes
                self.volume_mesh.add_scalar_quantity(quantity_name, cell_values, defined_on="cells")
        if particle_state.transport_state is not None:
            values = particle_state.transport_state.saturations.numpy()
            self.particle_cloud.add_scalar_quantity('Water saturation', values)
            samples = self.simulator.particle_samples
            cells = np.bincount(samples.source_tets, weights=samples.reference_volumes * values,
                                minlength=len(self.simulator.tet_mesh.tet_indices)) / self.simulator.tet_mesh.reference_volumes
            self.volume_mesh.add_scalar_quantity('Water saturation', cells, defined_on='cells')
        vertex_positions = self.simulator.vertex_positions.numpy()
        self.volume_mesh.update_vertex_positions(vertex_positions)
        relative_volumes = (ComputeSignedVolumes(vertex_positions, self.simulator.tet_mesh.tet_indices)
                            / self.simulator.tet_mesh.reference_volumes)
        self.volume_mesh.add_scalar_quantity("Mesh volume ratio", relative_volumes, defined_on="cells")

    def RunSteps(self, step_count):
        try:
            self.latest_stats = self.simulator.Advance(step_count)
            self.Refresh()
        except RuntimeError as error_value:
            self.last_error = str(error_value)
            self.is_running = False

    def Callback(self):
        _, self.is_running = self.psim.Checkbox("Run", self.is_running)
        _, self.frame_steps = self.psim.InputInt("Substeps per frame", self.frame_steps)
        self.frame_steps = max(1, min(self.frame_steps, 1000))
        if self.psim.Button("Single step") and not self.last_error:
            self.RunSteps(1)
        self.psim.SameLine()
        if self.psim.Button("Reset"):
            self.is_running = False
            self.last_error = ""
            self.latest_stats = None
            self.simulator.Reset()
            self.Refresh()
        if self.psim.Button("Add slice plane"):
            self.ps.add_scene_slice_plane()
        self.psim.SameLine()
        if self.psim.Button("Queue snapshot for exit") and not self.last_error:
            output_path = Path("outputs") / f"state_{self.simulator.step_count:07d}.npz"
            if getattr(self.simulator, "result_recorder", None) is not None:
                self.simulator.result_recorder.QueueSnapshot()
            else:
                self.simulator.SaveSnapshot(output_path)
        if self.is_running and not self.last_error:
            self.RunSteps(self.frame_steps)
        self.psim.TextUnformatted(f"Device: {self.simulator.device_name}")
        self.psim.TextUnformatted(f"Particles: {self.simulator.particle_data.particle_count}")
        self.psim.TextUnformatted(f"Step: {self.simulator.step_count}")
        stage = 'Settling' if self.simulator.IsSettling else 'Baking'
        self.psim.TextUnformatted(f"Stage: {stage}; baking time: {self.simulator.baking_time:.5f} s")
        if self.latest_stats:
            self.psim.TextUnformatted(f"Material volume: {self.latest_stats['particle_volume_m3']:.6g} m^3")
            self.psim.TextUnformatted(f"Time: {self.latest_stats['simulation_time']:.5f} s")
            self.psim.TextUnformatted(
                f"J: {self.latest_stats['minimum_j']:.4f} .. {self.latest_stats['maximum_j']:.4f}")
            self.psim.TextUnformatted(
                f"Inverted mesh tets: {self.latest_stats['inverted_mesh_tets']}")
            self.psim.TextUnformatted(
                f"Unmapped mesh vertices: {self.latest_stats['unmapped_mesh_vertices']}")
        material = self.simulator.material
        material_state = self.simulator.current_state.material_state
        if hasattr(material_state, 'porosities'):
            porosities = material_state.porosities.numpy()
            self.psim.TextUnformatted(f"Temperature: {material_state.temperatures.numpy().min():.2f} .. {material_state.temperatures.numpy().max():.2f} K")
            self.psim.TextUnformatted(f"Porosity: {porosities.min():.4f} .. {porosities.max():.4f}")
            self.psim.TextUnformatted(f"Yield stress: {material_state.yield_stresses.numpy()[0]:.4g} Pa")
        self.psim.TextUnformatted("Mesh reconstructed from current particle positions and F.")
        if self.last_error:
            self.psim.TextUnformatted("Stopped: " + self.last_error + ". Reset to continue.")

    def Show(self):
        self.Initialize()
        self.ps.show()
