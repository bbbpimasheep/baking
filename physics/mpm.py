from pathlib import Path
import math
import numpy as np
import warp as wp

from geometry.mesh_tracking import ParticleMeshBinding
from geometry.mesh_validation import ComputeSignedVolumes
from physics.quasistatic import SettlingMetrics
from physics.grid import GridState, UpdateGrid
from physics.materials.elastic import ElasticMaterial
from physics.materials.ding import DingMaterial
from physics.state import AllocateState, CreateParticleState
from physics.thermal_water import ThermalWaterTransport
from physics.transfer import ParticleToGrid, GridToParticle


class MpmSimulator:
    """Forward mode reuses buffers; history mode retains each step for wp.Tape.

    The renderer is outside the differentiable path. Maxwell/viscoplastic models
    can implement the material methods without modifying transfer kernels.
    """

    def __init__(self, tet_mesh, particle_samples, simulation_config, material_config,
                 initial_velocity=(0.0, 0.0, 0.0), retain_history=False, ding_config=None, transport_config=None):
        simulation_config.Validate(material_config)
        # Use Warp's user cache or WARP_CACHE_PATH, not the project directory.
        # Native CUDA file operations can fail on non-ASCII Windows paths.
        wp.init()
        device_name = simulation_config.device_name
        if device_name == "auto":
            device_name = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.device_name = str(wp.get_device(device_name))
        self.tet_mesh = tet_mesh
        self.particle_samples = particle_samples
        self.simulation_config = simulation_config
        self.material = ElasticMaterial(material_config)
        if material_config.model == 'ding2019':
            from config import DingConfig
            self.material = DingMaterial(material_config, ding_config or DingConfig(), self.device_name, retain_history)
        if isinstance(self.material, DingMaterial):
            self.material.simulation_config = simulation_config
        if isinstance(self.material, DingMaterial) and self.material.ding_config.gas_enabled and not (transport_config and transport_config.enabled):
            raise ValueError('Ding gas simulation requires transport.enabled=true; prescribed temperature schedules were removed')
        self.initial_velocity = initial_velocity
        self.retain_history = retain_history
        self.ValidateInitialDomain()
        self.mesh_binding = ParticleMeshBinding(tet_mesh, particle_samples, self.device_name)
        self.transport = None
        if transport_config is not None and transport_config.enabled:
            transport_config.Validate(material_config)
            self.material.initial_temperature_k = transport_config.initial_temperature_k
            self.transport = ThermalWaterTransport(self, transport_config)
        self.Reset()

    @property
    def settling_steps(self):
        if self.transport is None:
            return 0
        if self._baking_start_step is not None and self._baking_start_step > 0:
            return self._baking_start_step
        return int(math.ceil(self.transport.config.settling_duration_s / self.simulation_config.time_step - 1e-10))

    @property
    def IsSettling(self):
        return (self.transport is not None and self.transport.config.settling_duration_s > 0
                and self._baking_start_step is None)

    @property
    def baking_steps(self):
        return 0 if self.IsSettling else max(0, self.step_count - self.settling_steps)

    @property
    def baking_time(self):
        return self.baking_steps * self.simulation_config.time_step

    def UpdateSettling(self, old_state):
        config = self.transport.config
        time_step = self.simulation_config.time_step
        if not config.settling_auto:
            if self.step_count >= self.settling_steps:
                self._baking_start_step = self.step_count
                self.settling_reason = 'fixed_duration'
            return
        self.settling_metrics.zero_()
        wp.launch(SettlingMetrics, self.particle_data.particle_count, inputs=[
            self.current_state.particle_velocities, self.current_state.affine_velocities,
            self.settling_reference_plastic, self.current_state.material_state.plastic_gradients,
            self.settling_metrics, (self.step_count - self.settling_reference_step) * time_step], device=self.device_name)
        values = self.settling_metrics.numpy()
        # Average plastic evolution over a short window: differencing stored
        # float32 gradients at every tiny dt would magnify spectral roundoff.
        if (self.step_count - self.settling_reference_step) * time_step + 1e-12 >= min(.01, config.settling_hold_duration_s):
            self.settling_plastic_rate = float(values[2])
            wp.copy(self.settling_reference_plastic, self.current_state.material_state.plastic_gradients)
            self.settling_reference_step = self.step_count
        values[2] = self.settling_plastic_rate
        self.settling_diagnostics = dict(max_speed_m_s=float(values[0]),
            max_strain_rate_per_s=float(values[1]), max_plastic_rate_per_s=float(values[2]))
        eligible = (self.step_count - 1) * time_step + 1e-12 >= config.settling_min_duration_s
        small = values[0] <= config.settling_speed_tolerance and max(values[1:]) <= config.settling_rate_tolerance
        self.settling_quiet_steps = self.settling_quiet_steps + 1 if eligible and small else 0
        if self.settling_quiet_steps * time_step + 1e-12 >= config.settling_hold_duration_s:
            self._baking_start_step = self.step_count
            self.settling_reason = 'quasi_static'
        elif self.step_count >= self.settling_steps:
            self.settling_reason = 'timeout'
            raise RuntimeError('Settling did not reach quasi-static tolerances before settling_duration_s; '
                               'baking was not started. Increase the maximum duration or review tolerances/material.')

    def Masses(self, state=None):
        state = self.current_state if state is None else state
        return state.transport_state.particle_masses if state.transport_state is not None else self.particle_data.particle_masses

    def ValidateInitialDomain(self):
        simulation_config = self.simulation_config
        origin_vector = np.asarray(simulation_config.grid_origin)
        # Allow placement just above the nominal wall nodes, including near-contact meshes.
        # boundary_cells >= 2 still leaves a valid quadratic interpolation stencil.
        lower_bound = origin_vector + (simulation_config.boundary_cells - 1) * simulation_config.grid_spacing
        upper_bound = origin_vector + (simulation_config.grid_resolution -
                                      simulation_config.boundary_cells) * simulation_config.grid_spacing
        all_positions = np.concatenate([self.tet_mesh.vertex_positions,
                                        self.particle_samples.particle_positions])
        if np.any(all_positions < lower_bound) or np.any(all_positions > upper_bound):
            raise ValueError(f"model must initially fit inside {lower_bound} .. {upper_bound}; "
                             "adjust [mesh] mesh_scale/mesh_translation or [simulation] grid settings in the config file")

    def Reset(self):
        # Call Reset outside an active wp.Tape scope.
        if isinstance(self.material, DingMaterial):
            self.material.Reset()
        self.current_state, self.particle_data = CreateParticleState(
            self.particle_samples, self.device_name, self.initial_velocity, self.retain_history)
        if isinstance(self.material, DingMaterial):
            self.material.InitializeState(self.current_state)
        if self.transport is not None:
            self.transport.InitializeState(self.current_state)
        self.next_state = AllocateState(self.particle_data.particle_count, self.device_name)
        self.grid_state = GridState(self.simulation_config, self.device_name, self.retain_history)
        self.kirchhoff_stresses = wp.zeros(self.particle_data.particle_count, dtype=wp.mat33,
                                          device=self.device_name, requires_grad=self.retain_history)
        self.vertex_positions = wp.array(self.tet_mesh.vertex_positions, dtype=wp.vec3,
                                         device=self.device_name)
        self.error_flags = wp.zeros(2, dtype=int, device=self.device_name)
        self.state_history = [self.current_state] if self.retain_history else []
        self.workspace_history = []
        self.transport_workspace_history = []
        self.step_count = 0
        self._baking_start_step = None if self.transport is not None and self.transport.config.settling_duration_s > 0 else 0
        self.settling_quiet_steps = 0
        self.settling_reference_step = 0
        self.settling_plastic_rate = 0.0
        self.settling_reference_plastic = (wp.clone(self.current_state.material_state.plastic_gradients, requires_grad=False)
                                         if isinstance(self.material, DingMaterial) else None)
        self.settling_reason = 'pending' if self._baking_start_step is None else 'disabled'
        self.settling_diagnostics = dict(max_speed_m_s=0.0, max_strain_rate_per_s=0.0, max_plastic_rate_per_s=0.0)
        self.settling_metrics = wp.zeros(3, dtype=float, device=self.device_name)
        self.mesh_step_count = 0
        if getattr(self, "result_recorder", None) is not None:
            self.result_recorder.Reset()

    def Step(self):
        simulation_config = self.simulation_config
        was_settling = self.IsSettling
        original_input = self.current_state
        input_state = original_input
        if self.transport is not None:
            input_state = self.transport.Advance(input_state, simulation_config.time_step,
                                                 settling=was_settling)
        if self.retain_history:
            output_state = AllocateState(self.particle_data.particle_count, self.device_name, True)
            # Tape stores references, so each step needs independent intermediates.
            grid_state = GridState(simulation_config, self.device_name, True)
            stress_buffer = wp.zeros(self.particle_data.particle_count, dtype=wp.mat33,
                                     device=self.device_name, requires_grad=True)
        else:
            output_state = self.next_state
            grid_state = self.grid_state
            grid_state.Clear()
            stress_buffer = self.kirchhoff_stresses
        self.material.ComputeStress(input_state, stress_buffer, self.error_flags)
        if self.transport is not None:
            stress_buffer = self.transport.AddWaterStress(input_state, stress_buffer)
        grid_origin = wp.vec3(*simulation_config.grid_origin)
        wp.launch(ParticleToGrid, dim=self.particle_data.particle_count, inputs=[
            input_state.particle_positions, input_state.particle_velocities,
            input_state.affine_velocities, stress_buffer,
            self.Masses(input_state), self.particle_data.reference_volumes,
            grid_state.node_masses, grid_state.node_momenta,
            grid_origin, simulation_config.grid_spacing, simulation_config.grid_resolution,
            simulation_config.time_step, self.error_flags], device=self.device_name)
        wp.launch(UpdateGrid, dim=grid_state.node_count, inputs=[
            grid_state.node_masses, grid_state.node_momenta, grid_state.node_velocities,
            simulation_config.grid_resolution, simulation_config.boundary_cells,
            wp.vec3(*simulation_config.gravity), simulation_config.time_step], device=self.device_name)
        wp.launch(GridToParticle, dim=self.particle_data.particle_count, inputs=[
            input_state.particle_positions, input_state.deformation_gradients,
            grid_state.node_velocities, output_state.particle_positions,
            output_state.particle_velocities, output_state.affine_velocities,
            output_state.deformation_gradients, grid_origin, simulation_config.grid_spacing,
            simulation_config.grid_resolution, simulation_config.time_step,
            self.error_flags], device=self.device_name)
        self.material.AdvanceState(input_state, output_state, simulation_config.time_step)
        if isinstance(self.material, DingMaterial):
            # Reject an invalid pore state before accepting or recording this step.
            self.material.CheckState()
        if self.transport is not None:
            self.transport.FinishStep(input_state, output_state)
            if self.retain_history:
                self.transport_workspace_history.append(self.transport.workspace)
        if self.retain_history:
            self.state_history.append(output_state)
            self.workspace_history.append((grid_state, stress_buffer))
        self.current_state = output_state
        self.next_state = original_input
        self.grid_state = grid_state
        self.step_count += 1
        if was_settling:
            self.UpdateSettling(original_input)
        if getattr(self, "result_recorder", None) is not None:
            self.result_recorder.Record()
            if was_settling and not self.IsSettling:
                self.result_recorder.settled_measurement = dict(self.result_recorder.last_measurement)
                self.result_recorder.QueueSnapshot()

    def Advance(self, step_count):
        if step_count < 1:
            raise ValueError("step_count must be positive")
        for step_index in range(step_count):
            self.Step()
        return self.CheckState()

    def UpdateMesh(self, force_update=False):
        """Reconstruct per frame, outside the optimization tape.

        The latest particle state determines the mesh without vertex integration.
        """
        if force_update or self.mesh_step_count != self.step_count:
            self.mesh_binding.Reconstruct(self.current_state, self.vertex_positions)
            self.mesh_step_count = self.step_count

    def CheckState(self):
        error_flags = self.error_flags.numpy()
        if error_flags[0]:
            raise RuntimeError("nonpositive/nonfinite J; reduce dt or inspect loading and sampling")
        if error_flags[1]:
            raise RuntimeError("particle stencil left the grid; enlarge the domain or reduce dt")
        particle_positions = self.current_state.particle_positions.numpy()
        particle_velocities = self.current_state.particle_velocities.numpy()
        deformation_gradients = self.current_state.deformation_gradients.numpy()
        affine_velocities = self.current_state.affine_velocities.numpy()
        if not all(np.isfinite(state_array).all() for state_array in
                   (particle_positions, particle_velocities, deformation_gradients, affine_velocities)):
            raise RuntimeError("nonfinite particle state; simulation stopped")
        volume_ratios = np.linalg.det(deformation_gradients)
        if volume_ratios.min() <= 0.0:
            raise RuntimeError("inverted particle deformation; reduce dt or loading")
        grid_positions = ((particle_positions - np.array(self.simulation_config.grid_origin)) /
                          self.simulation_config.grid_spacing)
        if np.any(grid_positions < 0.5) or np.any(
                grid_positions >= self.simulation_config.grid_resolution - 1.5):
            raise RuntimeError("particle outside valid grid stencil")
        particle_masses = self.Masses().numpy().astype(np.float64)
        maximum_speed = float(np.linalg.norm(particle_velocities, axis=1).max())
        wave_speed = self.material.material_config.wave_speed
        if isinstance(self.material, DingMaterial) and self.material.ding_config.gas_enabled:
            wave_speed = max(wave_speed, self.material.GasWaveSpeed(self.current_state))
        if self.transport is not None and self.transport.config.water_enabled:
            # d(p_water)/d(log compression) = c_hat*s; add its acoustic stiffness.
            state = self.current_state
            water_wave_squared = (self.transport.config.water_stress * state.transport_state.saturations.numpy()
                                  * volume_ratios / state.material_state.reference_densities.numpy())
            wave_speed = float(np.sqrt(wave_speed * wave_speed + water_wave_squared.max()))
        cfl_value = ((maximum_speed + wave_speed) *
                     self.simulation_config.time_step / self.simulation_config.grid_spacing)
        if cfl_value > self.simulation_config.cfl_limit:
            raise RuntimeError(f"estimated CFL={cfl_value:.3f} > {self.simulation_config.cfl_limit}; reduce dt")
        if isinstance(self.material, DingMaterial):
            self.material.CheckState()
            material_state = self.current_state.material_state
            elastic_values = material_state.elastic_gradients.numpy()
            plastic_values = material_state.plastic_gradients.numpy()
            if not np.isfinite(elastic_values).all() or not np.isfinite(plastic_values).all():
                raise RuntimeError("nonfinite Ding internal state")
            if float(material_state.return_residuals.numpy().max()) > 1e-4:
                raise RuntimeError("Ding local return mapping did not converge")
        self.UpdateMesh()
        vertex_positions = self.vertex_positions.numpy()
        if not np.isfinite(vertex_positions).all():
            raise RuntimeError("nonfinite reconstructed mesh vertices")
        return {
            "step_count": self.step_count,
            "simulation_time": self.step_count * self.simulation_config.time_step,
            "particle_count": self.particle_data.particle_count,
            "particle_volume_m3": float(np.dot(self.particle_samples.reference_volumes, volume_ratios)),
            "total_mass": float(particle_masses.sum()),
            "grid_mass": float(self.grid_state.node_masses.numpy().sum(dtype=np.float64)),
            "minimum_j": float(volume_ratios.min()),
            "maximum_j": float(volume_ratios.max()),
            "maximum_speed": maximum_speed,
            "kinetic_energy": float(0.5 * np.sum(particle_masses *
                                                 np.sum(particle_velocities ** 2, axis=1))),
            "estimated_cfl": cfl_value,
            "unmapped_mesh_vertices": self.mesh_binding.unmapped_vertex_count,
            "inverted_mesh_tets": int(np.count_nonzero(ComputeSignedVolumes(
                vertex_positions, self.tet_mesh.tet_indices) <= 0.0)),
        }

    def CaptureSnapshot(self):
        self.UpdateMesh()
        snapshot_data = dict(
            particle_positions=self.current_state.particle_positions.numpy().copy(),
            particle_velocities=self.current_state.particle_velocities.numpy().copy(),
            deformation_gradients=self.current_state.deformation_gradients.numpy().copy(),
            affine_velocities=self.current_state.affine_velocities.numpy().copy(),
            particle_masses=self.Masses().numpy().copy(),
            reference_volumes=self.particle_data.reference_volumes.numpy().copy(),
            vertices=self.vertex_positions.numpy().copy(), tetrahedra=self.tet_mesh.tet_indices.copy(),
            reference_vertices=self.tet_mesh.vertex_positions.copy(),
            source_tets=self.particle_samples.source_tets.copy(),
            simulation_time=self.step_count * self.simulation_config.time_step,
            baking_time_s=self.baking_time, settling_steps=self.settling_steps)

        if isinstance(self.material, DingMaterial):
            for state_name in ('elastic_gradients', 'plastic_gradients', 'kirchhoff_stresses', 'return_residuals',
                               'porosities', 'temperatures', 'gelled', 'yield_stresses', 'viscosities',
                               'gas_moles', 'gas_pressures', 'porosity_residuals', 'reference_densities'):
                snapshot_data['ding_' + state_name] = getattr(self.current_state.material_state, state_name).numpy().copy()
        if self.current_state.transport_state is not None:
            from dataclasses import fields
            for field in fields(self.current_state.transport_state):
                snapshot_data["transport_" + field.name] = getattr(self.current_state.transport_state, field.name).numpy().copy()
        return snapshot_data

    def SaveSnapshot(self, output_path):
        self.UpdateMesh()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.CaptureSnapshot())
