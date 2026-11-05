"""Read-only measurements buffered in memory; files are written only at finish."""
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from geometry.mesh_validation import ComputeSignedVolumes


def ExtractSurface(vertex_positions, tet_indices):
    face_vertices = []
    opposite_vertices = []
    for opposite_index in range(4):
        local_face = [local_index for local_index in range(4) if local_index != opposite_index]
        face_vertices.append(tet_indices[:, local_face])
        opposite_vertices.append(tet_indices[:, opposite_index])
    all_faces = np.concatenate(face_vertices)
    all_opposites = np.concatenate(opposite_vertices)
    _, inverse_indices, face_counts = np.unique(np.sort(all_faces, axis=1), axis=0,
                                               return_inverse=True, return_counts=True)
    if np.any(face_counts > 2):
        raise ValueError('Nonmanifold tet mesh: more than two cells share a face')
    boundary_mask = face_counts[inverse_indices] == 1
    surface_faces = all_faces[boundary_mask].copy()
    boundary_positions = vertex_positions[surface_faces]
    inward_vectors = vertex_positions[all_opposites[boundary_mask]] - boundary_positions[:, 0]
    face_normals = np.cross(boundary_positions[:, 1] - boundary_positions[:, 0],
                            boundary_positions[:, 2] - boundary_positions[:, 0])
    flip_mask = np.einsum('ij,ij->i', face_normals, inward_vectors) > 0
    surface_faces[flip_mask, 1], surface_faces[flip_mask, 2] = (
        surface_faces[flip_mask, 2].copy(), surface_faces[flip_mask, 1].copy())
    surface_indices, remapped_faces = np.unique(surface_faces, return_inverse=True)
    return vertex_positions[surface_indices].copy(), remapped_faces.reshape(-1, 3)


class SimulationRecorder:
    def __init__(self, simulator, experiment_config):
        self.simulator = simulator
        self.experiment_config = experiment_config
        self.pending_snapshots = []
        self.finished_path = None
        self.Reset()

    def Measure(self):
        simulator = self.simulator
        simulator.UpdateMesh()
        vertex_positions = simulator.vertex_positions.numpy().astype(np.float64)
        deformation_gradients = simulator.current_state.deformation_gradients.numpy().astype(np.float64)
        if not np.isfinite(vertex_positions).all() or not np.isfinite(deformation_gradients).all():
            raise ValueError('Cannot record nonfinite simulation state')
        signed_volumes = ComputeSignedVolumes(vertex_positions, simulator.tet_mesh.tet_indices)
        volume_ratios = np.linalg.det(deformation_gradients)
        extent_values = np.ptp(vertex_positions, axis=0)
        material = simulator.material
        expansion_values = {}
        material_state = simulator.current_state.material_state
        if hasattr(material_state, 'porosities'):
            porosities = material_state.porosities.numpy()
            expansion_values.update(
                temperature_k=float(material_state.temperatures.numpy().mean()),
                porosity_min=float(porosities.min()), porosity_mean=float(porosities.mean(dtype=np.float64)),
                porosity_max=float(porosities.max()),
                yield_stress_pa=float(material_state.yield_stresses.numpy()[0]),
                viscosity_pa3_s=float(material_state.viscosities.numpy()[0]),
                gelled_fraction=float(material_state.gelled.numpy().mean()))
        if hasattr(material_state, 'gas_moles') and material.ding_config.gas_enabled:
            gas_moles = material_state.gas_moles.numpy()
            pressures = material_state.gas_pressures.numpy()
            expansion_values.update(
                gas_total_mol=float(np.dot(simulator.particle_samples.reference_volumes, gas_moles)),
                gas_pressure_min_pa=float(pressures.min()), gas_pressure_max_pa=float(pressures.max()),
                gas_pressure_mean_pa=float(pressures.mean()),
                pore_volume_m3=float(np.dot(simulator.particle_samples.reference_volumes, volume_ratios * porosities)))
        if simulator.current_state.transport_state is not None:
            state = simulator.current_state.transport_state
            temperature = state.temperatures.numpy()
            water = state.water_masses.numpy()
            saturation = state.saturations.numpy()
            surface_mask = np.zeros(len(water), dtype=bool)
            surface_mask[simulator.transport.owners.numpy()] = True
            positions = simulator.particle_samples.particle_positions
            center = np.mean(simulator.tet_mesh.vertex_positions, axis=0)
            radius = np.linalg.norm(positions - center, axis=1)
            core_mask = radius < 0.4 * radius.max()
            if not core_mask.any():
                core_mask = radius == radius.min()
            initial_energy = simulator.transport.config.initial_temperature_k * simulator.transport.capacities.numpy().sum()
            energy = float(np.dot(simulator.transport.capacities.numpy(), temperature))
            exchange = float(state.heat_exchanges.numpy().sum())
            losses = float(state.water_losses.numpy().sum())
            initial_water = (simulator.transport.config.water_density * simulator.transport.config.initial_saturation
                             * simulator.particle_samples.reference_volumes.sum())
            expansion_values.update(
                temperature_min_k=float(temperature.min()), temperature_max_k=float(temperature.max()),
                temperature_mean_k=float(np.average(temperature, weights=simulator.transport.capacities.numpy())),
                temperature_surface_k=float(temperature[surface_mask].mean()), temperature_core_k=float(temperature[core_mask].mean()),
                water_mass_kg=float(water.sum()), water_lost_kg=losses,
                water_balance_error_kg=float(water.sum() + losses - initial_water),
                saturation_min=float(saturation.min()), saturation_max=float(saturation.max()), saturation_mean=float(saturation.mean()),
                thermal_energy_j=energy, heat_exchange_j=exchange,
                energy_balance_error_j=float(energy - initial_energy - exchange))
        return dict(**expansion_values, baking_time_s=simulator.baking_time,
                    phase="settling" if simulator.IsSettling else "baking", step=simulator.step_count,
                    time_s=simulator.step_count * simulator.simulation_config.time_step,
                    length_x_m=float(extent_values[0]), width_z_m=float(extent_values[2]),
                    height_y_m=float(extent_values[1]),
                    particle_volume_m3=float(np.dot(simulator.particle_samples.reference_volumes, volume_ratios)),
                    mesh_volume_m3=float(np.abs(signed_volumes).sum()),
                    inverted_tets=int(np.count_nonzero(signed_volumes <= 0)),
                    invalid_particles=int(np.count_nonzero(volume_ratios <= 0)))

    def Reset(self):
        self.settled_measurement = None
        self.expansion_history = []
        self.height_history = []
        self.pending_snapshots = []
        self.initial_measurement = self.Measure()
        self.last_measurement = self.initial_measurement
        self.height_history.append((self.initial_measurement['time_s'], self.initial_measurement['height_y_m']))
        self.recorded_step = self.simulator.step_count
        self.expansion_history.append(dict(self.last_measurement))
        if self.experiment_config.run.snapshot_interval:
            self.QueueSnapshot()

    def Record(self):
        if self.recorded_step == self.simulator.step_count:
            return
        self.last_measurement = self.Measure()
        self.height_history.append((self.last_measurement['time_s'], self.last_measurement['height_y_m']))
        self.recorded_step = self.simulator.step_count
        self.expansion_history.append(dict(self.last_measurement))
        interval = self.experiment_config.run.snapshot_interval
        if interval and self.simulator.step_count % interval == 0:
            self.QueueSnapshot()

    def QueueSnapshot(self):
        self.pending_snapshots.append((self.simulator.step_count, self.simulator.CaptureSnapshot()))

    def Finish(self, status='completed', error_message='', console_text=''):
        if self.finished_path is not None:
            return self.finished_path
        # A failed step may be invalid. Keep the last valid measurement clearly labelled.
        try:
            self.Record()
        except (ValueError, RuntimeError) as record_error:
            status = 'failed'
            error_message = (error_message + '; ' + str(record_error)).strip('; ')
        output_root = Path(self.experiment_config.run.log_path).expanduser()
        run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8]
        run_path = output_root / run_name
        run_path.mkdir(parents=True, exist_ok=False)
        simulator = self.simulator
        final_measurement = self.last_measurement
        summary_data = dict(
            status=status, error=error_message,
            units=dict(length='m', volume='m^3', time='s', mass='kg', stress='Pa', temperature='K', porosity='1', ding_viscosity='Pa^3 s', gas_content='mol/m^3 of reference volume', gas_amount='mol'),
            conventions=dict(dimensions='axis-aligned reconstructed mesh bounds: length=x, width=z, height=y',
                             particle_volume='sum(reference_volume * det(F))',
                             mesh_volume='sum(abs(signed tetrahedron volume)); invalid as physical volume if inverted/overlapping',
                             surface='boundary of original tet connectivity at final reconstructed vertices'),
            initial=self.initial_measurement, settled=self.settled_measurement, final=final_measurement,
            settling_steps=simulator.settling_steps if not simulator.IsSettling else simulator.step_count,
            settling_reason=simulator.settling_reason, settling_diagnostics=simulator.settling_diagnostics,
            actual_settling_duration_s=(simulator.step_count if simulator.IsSettling else simulator.settling_steps) * simulator.simulation_config.time_step,
            attempted_step=simulator.step_count, parameters=asdict(self.experiment_config),
            actual_device=simulator.device_name, sample_count=len(self.height_history),
            total_mass_kg=float(simulator.Masses().numpy().sum(dtype=np.float64)))
        if final_measurement['step'] == simulator.step_count:
            surface_vertices, surface_faces = ExtractSurface(simulator.vertex_positions.numpy(), simulator.tet_mesh.tet_indices)
            with (run_path / 'final_surface.obj').open('w', encoding='ascii') as surface_file:
                surface_file.write('# Units: metres. Boundary of the reconstructed tetrahedral mesh.\n')
                for vertex_position in surface_vertices:
                    surface_file.write('v ' + ' '.join(format(float(value), '.9g') for value in vertex_position) + '\n')
                for face_indices in surface_faces:
                    surface_file.write('f ' + ' '.join(str(int(value) + 1) for value in face_indices) + '\n')
            simulator.SaveSnapshot(run_path / 'final_state.npz')
        else:
            summary_data['surface_omitted'] = 'Final state invalid; measurements refer to last valid recorded state'
        with (run_path / 'height_history.csv').open('w', newline='', encoding='ascii') as history_file:
            csv_writer = csv.writer(history_file)
            csv_writer.writerow(['time_s', 'height_y_m'])
            csv_writer.writerows(self.height_history)
        self.SaveHeightPlot(run_path / 'height_history.svg')
        saved_height_history = self.height_history
        self.height_history = [(entry['time_s'], entry['particle_volume_m3']) for entry in self.expansion_history]
        self.SaveHeightPlot(run_path / 'volume_history.svg', 'Material volume (m^3)', 'Material volume over time')
        self.height_history = saved_height_history
        if self.experiment_config.material.model == 'ding2019' and self.experiment_config.ding.gas_enabled:
            for field_name, file_name, label_name, title_name in (
                ('gas_total_mol', 'gas_amount_history.svg', 'Gas amount (mol)', 'Gas amount over time'),
                ('gas_pressure_mean_pa', 'gas_pressure_history.svg', 'Mean pore pressure (Pa)', 'Absolute pore pressure over time'),
                ('temperature_k', 'temperature_history.svg', 'Mean temperature (K)', 'Mean temperature over time')):
                self.height_history = [(entry['time_s'], entry[field_name]) for entry in self.expansion_history]
                self.SaveHeightPlot(run_path / file_name, label_name, title_name)
            self.height_history = saved_height_history
        if simulator.transport is not None:
            for field, filename, label in (
                ('temperature_surface_k', 'surface_temperature.svg', 'Surface-owner temperature (K)'),
                ('temperature_core_k', 'core_temperature.svg', 'Initial core temperature (K)'),
                ('water_mass_kg', 'water_mass.svg', 'Water mass (kg)'),
                ('saturation_mean', 'water_saturation.svg', 'Mean water saturation (1)')):
                self.height_history = [(entry['time_s'], entry[field]) for entry in self.expansion_history]
                self.SaveHeightPlot(run_path / filename, label, label + ' over time')
            self.height_history = saved_height_history
            thermal_fields = ['time_s', 'baking_time_s', 'phase'] + [field for field in self.expansion_history[0]
                if field.startswith(('temperature_', 'water_', 'saturation_', 'thermal_', 'heat_', 'energy_'))]
            with (run_path / 'thermal_water_history.csv').open('w', newline='', encoding='ascii') as thermal_file:
                writer = csv.DictWriter(thermal_file, fieldnames=thermal_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.expansion_history)
        with (run_path / 'expansion_history.csv').open('w', newline='', encoding='ascii') as expansion_file:
            expansion_writer = csv.DictWriter(expansion_file, fieldnames=list(self.expansion_history[0]))
            expansion_writer.writeheader()
            expansion_writer.writerows(self.expansion_history)
        if self.experiment_config.material.model == 'ding2019':
            with (run_path / 'material_history.csv').open('w', newline='', encoding='ascii') as material_file:
                field_names = ['time_s', 'temperature_k', 'porosity_min', 'porosity_mean', 'porosity_max',
                               'yield_stress_pa', 'viscosity_pa3_s', 'gelled_fraction']
                if self.experiment_config.ding.gas_enabled:
                    field_names += ['gas_total_mol', 'gas_pressure_min_pa', 'gas_pressure_max_pa',
                                    'gas_pressure_mean_pa', 'pore_volume_m3']
                material_writer = csv.DictWriter(material_file, fieldnames=field_names, extrasaction='ignore')
                material_writer.writeheader()
                material_writer.writerows(self.expansion_history)
        model_note = 'Neo-Hookean elasticity'
        if self.experiment_config.material.model == 'ding2019':
            model_note = 'Ding solid skeleton with particle temperature and plastic-volume porosity'
        if simulator.transport is not None:
            model_note = ('Ding Sections 3.2/4/5.5: local heat, water diffusion/loss and saturation stress; '
                          'conservative grid finite-volume fluxes; fixed initial heat capacity; geometric tray contact; '
                          'c=0; no relative-water momentum/enthalpy, vapor transport, radiation, fracture or gas diffusion; '
                          'optional latent cooling is an added approximation, not a paper equation')
        summary_data['model_note'] = model_note
        for snapshot_step, snapshot_data in self.pending_snapshots:
            np.savez_compressed(run_path / f'snapshot_{snapshot_step:07d}.npz', **snapshot_data)
        (run_path / 'summary.json').write_text(json.dumps(summary_data, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
        (run_path / 'console.log').write_text(console_text, encoding='utf-8')
        self.finished_path = run_path
        return run_path

    def SaveHeightPlot(self, plot_path, quantity_name='Height (m), y axis', plot_title='Reconstructed material height over time'):
        history_values = np.asarray(self.height_history)
        time_values, height_values = history_values[:, 0], history_values[:, 1]
        time_min, time_max = 0.0, max(float(time_values.max()), self.simulator.simulation_config.time_step)
        height_min, height_max = float(height_values.min()), float(height_values.max())
        padding_value = max((height_max - height_min) * 0.1, abs(height_max) * 0.01, 1e-6)
        height_min -= padding_value
        height_max += padding_value
        point_values = [f'{70 + 680 * (time_value-time_min)/(time_max-time_min):.3f},{350 - 300 * (height_value-height_min)/(height_max-height_min):.3f}'
                        for time_value, height_value in self.height_history]
        plot_text = f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420"><rect width="800" height="420" fill="white"/><g fill="black" font-family="sans-serif" font-size="14"><text x="260" y="25">{plot_title}</text><text x="350" y="405">Time (s)</text><text x="15" y="210" transform="rotate(-90 15 210)">{quantity_name}</text><text x="70" y="375">0</text><text x="690" y="375">{time_max:.5g}</text><text x="25" y="350">{height_min:.4g}</text><text x="25" y="55">{height_max:.4g}</text></g><path d="M70 50 V350 H750" fill="none" stroke="black"/><polyline points="{" ".join(point_values)}" fill="none" stroke="#2563eb" stroke-width="2"/></svg>'
        plot_path.write_text(plot_text, encoding='ascii')
