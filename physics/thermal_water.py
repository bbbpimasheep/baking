"""Ding 2019 Sections 3.2, 4, 5.5 with conservative scalar MPM transfers.

Numerical choices: six-neighbor symmetric finite-volume grid fluxes instead of
the paper's Galerkin diffusion; exact original-mesh boundary triangles instead
of closest particles on empty-grid boundaries. Heat uses fixed initial effective
capacity. Relative-water momentum/enthalpy transport and thermoelastic c are
omitted. Optional latent cooling is an explicit addition, not in Eq. (9).

Paper equations (SI): s=m_w/(rho_water*J*V0), q_w=-K*T*rho_water*s*grad(s)
(Sections 3.2, Eq. 8), q_w.n=beta_w*T*s, rho*alpha*Tdot=div(kappa*grad(T))
(Eq. 9 with c=0), sigma_w=-c_hat*(s-s0)*I (Eq. 14). Particle transfers use
the conservative relative-mass Eq. (19) and FLIP temperature Eq. (21).
Here mw is conserved on moving material volumes; geometric dilution is included
once through J in s, rather than adding Eq. (25)'s old-volume density dilation
term to a material mass. This is a conservative split, not the exact paper stencil.

Heat conductance for occupied neighbors is kappa*dx*f_ij, with symmetric harmonic
occupied-volume fraction f_ij=2*Vi*Vj/(Vi+Vj)/dx^3. Water conductance is
K*rho_water*dx*f_ij*mean(T)*mean(s). Pairwise fluxes cancel globally.
Boundary heat is h*S*(T_oven-Tp); this adopts Fourier's physical hot-to-cold sign
despite the inconsistent signs in the paper's printed Robin/discrete heat forms.
Optional evaporation cooling is -L*beta_w*Tp*sp*S (W). No vapor is stored.
Capacity is initial mixture mass times configured effective specific_heat (J/K),
fixed during drying: sensible enthalpy carried by diffusing/lost water is omitted.
These approximations suit a workflow test with small water redistribution/loss.

One explicit split: geometry/surface -> scalar P2G -> pairwise diffusion and
Robin flux -> scalar G2P -> local chemical source -> gas and water stress ->
mechanical MPM -> solid plastic state/porosity -> accepted pressure/saturation.
Face topology/ownership and contact classification are discrete. Derivatives are
piecewise valid away from contact, support, yield and gelatinization switches.
All scalar intermediates are retained by MpmSimulator in differentiation mode.
"""
from dataclasses import dataclass, replace
import numpy as np
import warp as wp
from physics.transfer import quadratic_weight, stencil_base, flatten_node
from physics.materials.gas import gas_constant, gas_pressure


@dataclass
class TransportState:
    temperatures: object
    water_masses: object
    saturations: object
    water_losses: object
    heat_exchanges: object
    particle_masses: object


def BoundaryFaces(tet_mesh, samples):
    candidates = {}
    for tet_index, tet in enumerate(tet_mesh.tet_indices):
        for opposite in range(4):
            face = np.delete(tet, opposite).copy()
            key = tuple(sorted(face))
            if key in candidates:
                candidates[key] = None
            else:
                points = tet_mesh.vertex_positions[face]
                if np.dot(np.cross(points[1] - points[0], points[2] - points[0]),
                          tet_mesh.vertex_positions[tet[opposite]] - points[0]) > 0:
                    face[[1, 2]] = face[[2, 1]]
                candidates[key] = (face, tet_index)
    faces, owners = [], []
    for entry in candidates.values():
        if entry is not None:
            face, tet_index = entry
            particle_indices = np.flatnonzero(samples.source_tets == tet_index)
            center = tet_mesh.vertex_positions[face].mean(axis=0)
            owner = particle_indices[np.argmin(np.linalg.norm(samples.particle_positions[particle_indices] - center, axis=1))]
            faces.append(face)
            owners.append(owner)
    return np.asarray(faces, dtype=np.int32), np.asarray(owners, dtype=np.int32)


@wp.kernel
def SurfaceAreas(vertices: wp.array(dtype=wp.vec3), faces: wp.array(dtype=wp.vec3i), owners: wp.array(dtype=int),
                 exposed_areas: wp.array(dtype=wp.float64), contact_areas: wp.array(dtype=wp.float64),
                 floor_height: float, contact_distance: float):
    index = wp.tid()
    face = faces[index]
    first = wp.vec3d(vertices[face[0]])
    second = wp.vec3d(vertices[face[1]])
    third = wp.vec3d(vertices[face[2]])
    normal = wp.cross(second - first, third - first)
    area = wp.float64(0.5) * wp.length(normal)
    center_y = (first[1] + second[1] + third[1]) / wp.float64(3.0)
    if center_y <= wp.float64(floor_height + contact_distance) and normal[1] < -wp.float64(0.2) * wp.length(normal):
        wp.atomic_add(contact_areas, owners[index], area)
    else:
        wp.atomic_add(exposed_areas, owners[index], area)


@wp.kernel
def ScalarToGrid(positions: wp.array(dtype=wp.vec3), gradients: wp.array(dtype=wp.mat33),
                 reference_volumes: wp.array(dtype=float), capacities: wp.array(dtype=wp.float64),
                 temperatures: wp.array(dtype=wp.float64), water_masses: wp.array(dtype=wp.float64),
                 exposed: wp.array(dtype=wp.float64), contact: wp.array(dtype=wp.float64),
                 heat_parameter: wp.array(dtype=float), oven_temperature: float, tray_temperature: float,
                 tray_transfer: float, surface_loss: float, water_density: float, latent_heat: float,
                 water_enabled: int, node_volumes: wp.array(dtype=wp.float64), node_capacities: wp.array(dtype=wp.float64),
                 node_energy: wp.array(dtype=wp.float64), node_water: wp.array(dtype=wp.float64),
                 node_heat: wp.array(dtype=wp.float64), node_loss: wp.array(dtype=wp.float64),
                 node_exchange: wp.array(dtype=wp.float64), particle_heat: wp.array(dtype=wp.float64),
                 particle_loss: wp.array(dtype=wp.float64), origin: wp.vec3, spacing: float, resolution: int):
    index = wp.tid()
    volume = wp.float64(reference_volumes[index]) * wp.float64(wp.determinant(gradients[index]))
    temperature = temperatures[index]
    saturation = water_masses[index] / (wp.float64(water_density) * volume)
    loss = wp.float64(0.0)
    if water_enabled != 0:
        loss = wp.float64(surface_loss) * temperature * saturation * exposed[index]
    heat = (wp.float64(heat_parameter[0]) * exposed[index] * (wp.float64(oven_temperature) - temperature)
            + wp.float64(tray_transfer) * contact[index] * (wp.float64(tray_temperature) - temperature)
            - wp.float64(latent_heat) * loss)
    exchange = wp.float64(heat_parameter[0]) * exposed[index] + wp.float64(tray_transfer) * contact[index]
    if water_enabled != 0:
        # Include the explicit cooling slope in the heat stability bound.
        exchange += wp.float64(latent_heat * surface_loss) * saturation * exposed[index]
    particle_heat[index] = heat
    particle_loss[index] = loss
    grid_position = (positions[index] - origin) / spacing
    base = stencil_base(grid_position)
    local = grid_position - wp.vec3(base)
    for x_offset in range(3):
        for y_offset in range(3):
            for z_offset in range(3):
                weight = wp.float64(quadratic_weight(local[0], x_offset) * quadratic_weight(local[1], y_offset) * quadratic_weight(local[2], z_offset))
                node = flatten_node(base + wp.vec3i(x_offset, y_offset, z_offset), resolution)
                wp.atomic_add(node_volumes, node, weight * volume)
                wp.atomic_add(node_capacities, node, weight * capacities[index])
                wp.atomic_add(node_energy, node, weight * capacities[index] * temperature)
                wp.atomic_add(node_water, node, weight * water_masses[index])
                wp.atomic_add(node_heat, node, weight * heat)
                wp.atomic_add(node_loss, node, weight * loss)
                wp.atomic_add(node_exchange, node, weight * exchange)


@wp.kernel
def GridFields(volumes: wp.array(dtype=wp.float64), capacities: wp.array(dtype=wp.float64),
               energy: wp.array(dtype=wp.float64), water: wp.array(dtype=wp.float64),
               temperatures: wp.array(dtype=wp.float64), saturations: wp.array(dtype=wp.float64), water_density: float):
    index = wp.tid()
    if capacities[index] > wp.float64(0.0):
        temperatures[index] = energy[index] / capacities[index]
        saturations[index] = water[index] / (wp.float64(water_density) * volumes[index])


@wp.kernel
def DiffuseGrid(volumes: wp.array(dtype=wp.float64), capacities: wp.array(dtype=wp.float64),
                temperatures: wp.array(dtype=wp.float64), saturations: wp.array(dtype=wp.float64),
                water: wp.array(dtype=wp.float64), boundary_heat: wp.array(dtype=wp.float64),
                boundary_loss: wp.array(dtype=wp.float64), boundary_exchange: wp.array(dtype=wp.float64),
                temperature_changes: wp.array(dtype=wp.float64), water_changes: wp.array(dtype=wp.float64),
                errors: wp.array(dtype=int), conductivity: float, diffusivity: float, water_density: float,
                water_enabled: int, spacing: float, resolution: int, time_step: float):
    index = wp.tid()
    if capacities[index] <= wp.float64(0.0):
        return
    coord = wp.vec3i(index // (resolution * resolution), (index // resolution) % resolution, index % resolution)
    heat_rate = boundary_heat[index]
    water_rate = -boundary_loss[index]
    heat_diagonal = boundary_exchange[index]
    water_diagonal = wp.float64(0.0)
    for axis in range(3):
        for side in range(2):
            other_coord = coord
            other_coord[axis] = other_coord[axis] + 2 * side - 1
            if other_coord[axis] >= 0 and other_coord[axis] < resolution:
                other = flatten_node(other_coord, resolution)
                if capacities[other] > wp.float64(0.0):
                    # Symmetric occupied-volume face fraction; both endpoints use identical flux coefficients.
                    fill = wp.float64(2.0) * volumes[index] * volumes[other] / (volumes[index] + volumes[other]) / wp.float64(spacing * spacing * spacing)
                    heat_coefficient = wp.float64(conductivity * spacing) * fill
                    heat_rate += heat_coefficient * (temperatures[other] - temperatures[index])
                    heat_diagonal += heat_coefficient
                    if water_enabled != 0:
                        mobility = (wp.float64(diffusivity * water_density * spacing) * fill
                            * wp.float64(0.5) * (temperatures[index] + temperatures[other])
                            * wp.float64(0.5) * (saturations[index] + saturations[other]))
                        water_rate += mobility * (saturations[other] - saturations[index])
                        water_diagonal += mobility / (wp.float64(water_density) * volumes[index])
    if wp.float64(time_step) * heat_diagonal > capacities[index]:
        wp.atomic_max(errors, 0, 1)
    change = wp.float64(time_step) * water_rate
    if water[index] + change < wp.float64(0.0) or wp.float64(time_step) * water_diagonal > wp.float64(1.0):
        wp.atomic_max(errors, 1, 1)
    temperature_changes[index] = wp.float64(time_step) * heat_rate / capacities[index]
    water_changes[index] = change


@wp.kernel
def ScalarToParticles(positions: wp.array(dtype=wp.vec3), old_temperatures: wp.array(dtype=wp.float64),
                      old_water: wp.array(dtype=wp.float64), old_losses: wp.array(dtype=wp.float64),
                      old_exchanges: wp.array(dtype=wp.float64), dry_masses: wp.array(dtype=wp.float64),
                      reference_volumes: wp.array(dtype=float), gradients: wp.array(dtype=wp.mat33),
                      node_water: wp.array(dtype=wp.float64), temperature_changes: wp.array(dtype=wp.float64),
                      water_changes: wp.array(dtype=wp.float64), particle_heat: wp.array(dtype=wp.float64),
                      particle_loss: wp.array(dtype=wp.float64), new_temperatures: wp.array(dtype=wp.float64),
                      material_temperatures: wp.array(dtype=float), new_water: wp.array(dtype=wp.float64),
                      new_losses: wp.array(dtype=wp.float64), new_exchanges: wp.array(dtype=wp.float64),
                      new_masses: wp.array(dtype=float), densities: wp.array(dtype=float),
                      saturations: wp.array(dtype=wp.float64), water_density: float,
                      origin: wp.vec3, spacing: float, resolution: int, time_step: float):
    index = wp.tid()
    grid_position = (positions[index] - origin) / spacing
    base = stencil_base(grid_position)
    local = grid_position - wp.vec3(base)
    temperature_change = wp.float64(0.0)
    mass_ratio_change = wp.float64(0.0)
    for x_offset in range(3):
        for y_offset in range(3):
            for z_offset in range(3):
                weight = wp.float64(quadratic_weight(local[0], x_offset) * quadratic_weight(local[1], y_offset) * quadratic_weight(local[2], z_offset))
                node = flatten_node(base + wp.vec3i(x_offset, y_offset, z_offset), resolution)
                temperature_change += weight * temperature_changes[node]
                if node_water[node] > wp.float64(0.0):
                    mass_ratio_change += weight * water_changes[node] / node_water[node]
    temperature = old_temperatures[index] + temperature_change
    mass = old_water[index] * (wp.float64(1.0) + mass_ratio_change)
    new_temperatures[index] = temperature
    material_temperatures[index] = float(temperature)
    new_water[index] = mass
    new_losses[index] = old_losses[index] + wp.float64(time_step) * particle_loss[index]
    new_exchanges[index] = old_exchanges[index] + wp.float64(time_step) * particle_heat[index]
    new_masses[index] = float(dry_masses[index] + mass)
    densities[index] = float((dry_masses[index] + mass) / wp.float64(reference_volumes[index]))
    saturations[index] = mass / (wp.float64(water_density) * wp.float64(reference_volumes[index]) * wp.float64(wp.determinant(gradients[index])))


@wp.kernel
def LocalGas(old_moles: wp.array(dtype=wp.float64), old_temperatures: wp.array(dtype=wp.float64),
             new_temperatures: wp.array(dtype=wp.float64), new_moles: wp.array(dtype=wp.float64),
             pressures: wp.array(dtype=wp.float64), gradients: wp.array(dtype=wp.mat33),
             porosities: wp.array(dtype=float), parameter: wp.array(dtype=float), density: float,
             peak: float, width: float, time_step: float, produce: int):
    index = wp.tid()
    moles = old_moles[index]
    if produce != 0:
        temperature = wp.float64(0.5) * (old_temperatures[index] + new_temperatures[index])
        moles += wp.float64(time_step * density) * wp.float64(parameter[0]) * wp.exp(-((temperature - wp.float64(peak)) / wp.float64(width)) ** wp.float64(2.0))
    new_moles[index] = moles
    if porosities[index] > 0.0:
        pressures[index] = gas_pressure(moles, float(new_temperatures[index]), wp.determinant(gradients[index]), porosities[index])


@wp.kernel
def WaterStress(gradients: wp.array(dtype=wp.mat33), saturations: wp.array(dtype=wp.float64),
                old_stress: wp.array(dtype=wp.mat33), new_stress: wp.array(dtype=wp.mat33),
                initial_saturation: float, stiffness: float):
    index = wp.tid()
    new_stress[index] = old_stress[index] - wp.determinant(gradients[index]) * stiffness * (float(saturations[index]) - initial_saturation) * wp.identity(n=3, dtype=float)


@wp.kernel
def Saturations(water: wp.array(dtype=wp.float64), reference_volumes: wp.array(dtype=float),
                gradients: wp.array(dtype=wp.mat33), saturations: wp.array(dtype=wp.float64), density: float):
    index = wp.tid()
    saturations[index] = water[index] / (wp.float64(density) * wp.float64(reference_volumes[index]) * wp.float64(wp.determinant(gradients[index])))


class ThermalWaterTransport:
    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self.device_name = simulator.device_name
        self.requires_grad = simulator.retain_history
        samples = simulator.particle_samples
        density = simulator.material.material_config.density
        self.reference_volumes = wp.array(samples.reference_volumes, dtype=float, device=self.device_name)
        self.dry_masses = wp.array((density - config.water_density * config.initial_saturation) * samples.reference_volumes,
                                  dtype=wp.float64, device=self.device_name)
        self.capacities = wp.array(density * config.specific_heat * samples.reference_volumes, dtype=wp.float64, device=self.device_name)
        self.heat_transfer_parameter = wp.array([config.heat_transfer], dtype=float, device=self.device_name, requires_grad=self.requires_grad)
        faces, owners = BoundaryFaces(simulator.tet_mesh, samples)
        self.faces = wp.array(faces, dtype=wp.vec3i, device=self.device_name)
        self.owners = wp.array(owners, dtype=int, device=self.device_name)
        self.error_flags = wp.zeros(2, dtype=int, device=self.device_name)
        self.workspace = None

    def Array(self, count, dtype=wp.float64):
        return wp.zeros(count, dtype=dtype, device=self.device_name, requires_grad=self.requires_grad)

    def InitializeState(self, state):
        count = len(state.particle_positions)
        state.transport_state = TransportState(
            wp.full(count, self.config.initial_temperature_k, dtype=wp.float64, device=self.device_name, requires_grad=self.requires_grad),
            wp.array(self.config.water_density * self.config.initial_saturation * self.simulator.particle_samples.reference_volumes,
                     dtype=wp.float64, device=self.device_name, requires_grad=self.requires_grad),
            wp.full(count, self.config.initial_saturation, dtype=wp.float64, device=self.device_name, requires_grad=self.requires_grad),
            self.Array(count), self.Array(count),
            wp.array(self.simulator.particle_samples.particle_masses, dtype=float, device=self.device_name, requires_grad=self.requires_grad))
        self.error_flags.zero_()

    def Advance(self, state, time_step, settling=False):
        config = self.config
        simulation = self.simulator.simulation_config
        count = len(state.particle_positions)
        node_count = simulation.grid_resolution ** 3
        class Workspace:
            pass
        work = Workspace()
        # Preserve all mechanical/internal state. Only scalar evolution and gas
        # production pause during the prescribed isothermal resting interval.
        work.heat_parameter = self.Array(1, float) if settling else self.heat_transfer_parameter
        work.vertices = self.Array(len(self.simulator.tet_mesh.vertex_positions), wp.vec3)
        self.simulator.mesh_binding.Reconstruct(state, work.vertices)
        work.exposed = self.Array(count)
        work.contact = self.Array(count)
        floor = simulation.grid_origin[1] + (simulation.boundary_cells - 1) * simulation.grid_spacing
        wp.launch(SurfaceAreas, len(self.faces), inputs=[work.vertices, self.faces, self.owners, work.exposed, work.contact,
            floor, config.contact_distance], device=self.device_name)
        for name in ('volumes', 'capacities', 'energy', 'water', 'heat', 'loss', 'exchange', 'temperatures', 'saturations', 'temperature_changes', 'water_changes'):
            setattr(work, name, self.Array(node_count))
        work.particle_heat = self.Array(count)
        work.particle_loss = self.Array(count)
        old = state.transport_state
        output = TransportState(self.Array(count), self.Array(count), self.Array(count), self.Array(count), self.Array(count), self.Array(count, float))
        material_temperatures = self.Array(count, float)
        reference_densities = self.Array(count, float)
        origin = wp.vec3(*simulation.grid_origin)
        wp.launch(ScalarToGrid, count, inputs=[state.particle_positions, state.deformation_gradients, self.reference_volumes,
            self.capacities, old.temperatures, old.water_masses, work.exposed, work.contact, work.heat_parameter,
            config.oven_temperature_k, config.tray_temperature_k, 0.0 if settling else config.tray_heat_transfer,
            0.0 if settling else config.surface_water_loss, config.water_density, config.latent_heat, int(config.water_enabled),
            work.volumes, work.capacities, work.energy, work.water, work.heat, work.loss, work.exchange,
            work.particle_heat, work.particle_loss, origin, simulation.grid_spacing, simulation.grid_resolution], device=self.device_name)
        wp.launch(GridFields, node_count, inputs=[work.volumes, work.capacities, work.energy, work.water,
            work.temperatures, work.saturations, config.water_density], device=self.device_name)
        wp.launch(DiffuseGrid, node_count, inputs=[work.volumes, work.capacities, work.temperatures, work.saturations,
            work.water, work.heat, work.loss, work.exchange, work.temperature_changes, work.water_changes,
            self.error_flags, 0.0 if settling else config.conductivity, 0.0 if settling else config.water_diffusivity, config.water_density, int(config.water_enabled),
            simulation.grid_spacing, simulation.grid_resolution, time_step], device=self.device_name)
        flags = self.error_flags.numpy()
        if flags[0] or flags[1]:
            raise RuntimeError('Heat/water explicit stability bound exceeded; reduce time_step; no temperature or water clipping is applied')
        wp.launch(ScalarToParticles, count, inputs=[state.particle_positions, old.temperatures, old.water_masses,
            old.water_losses, old.heat_exchanges, self.dry_masses, self.reference_volumes, state.deformation_gradients,
            work.water, work.temperature_changes, work.water_changes, work.particle_heat, work.particle_loss,
            output.temperatures, material_temperatures, output.water_masses, output.water_losses,
            output.heat_exchanges, output.particle_masses, reference_densities, output.saturations, config.water_density,
            origin, simulation.grid_spacing, simulation.grid_resolution, time_step], device=self.device_name)
        material_state = replace(state.material_state, temperatures=material_temperatures, reference_densities=reference_densities)
        material = self.simulator.material
        if material.ding_config.gas_enabled:
            material_state.gas_moles = self.Array(count)
            material_state.gas_pressures = self.Array(count)
            wp.launch(LocalGas, count, inputs=[state.material_state.gas_moles, old.temperatures, output.temperatures,
                material_state.gas_moles, material_state.gas_pressures, state.deformation_gradients, state.material_state.porosities,
                material.production_parameter, material.material_config.density, material.ding_config.gas_reaction_temperature_k,
                material.ding_config.gas_reaction_width_k, time_step, int(not settling)], device=self.device_name)
        work.prepared = replace(state, material_state=material_state, transport_state=output)
        self.workspace = work
        return work.prepared

    def AddWaterStress(self, state, stress):
        output = self.Array(len(state.particle_positions), wp.mat33)
        wp.launch(WaterStress, len(output), inputs=[state.deformation_gradients, state.transport_state.saturations,
            stress, output, self.config.initial_saturation, self.config.water_stress if self.config.water_enabled else 0.0], device=self.device_name)
        self.workspace.stress_before_water = stress
        return output

    def FinishStep(self, prepared, output):
        output.transport_state = replace(prepared.transport_state, saturations=self.Array(len(output.particle_positions)))
        wp.launch(Saturations, len(output.particle_positions), inputs=[output.transport_state.water_masses,
            self.reference_volumes, output.deformation_gradients, output.transport_state.saturations, self.config.water_density], device=self.device_name)
