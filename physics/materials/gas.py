"""Ding 2019 Section 5.4: reference-volume gas content and ideal-gas stress.

Numerical additions: ambient boundary traction,
gas acoustic CFL check. No diffusion, depletion, gas mass in inertia or heat solve. Production is evaluated from particle temperatures by transport.
"""
import warp as wp

gas_constant = 8.31446261815324


@wp.func
def gas_pressure(moles: wp.float64, temperature: float, volume_ratio: float, porosity: float):
    return moles * wp.float64(gas_constant) * wp.float64(temperature) / (wp.float64(volume_ratio) * wp.float64(porosity))


@wp.kernel
def CombineGasStress(solid_stresses: wp.array(dtype=wp.mat33), gradients: wp.array(dtype=wp.mat33),
                     porosities: wp.array(dtype=float), moles: wp.array(dtype=wp.float64),
                     temperatures: wp.array(dtype=float), total_stresses: wp.array(dtype=wp.mat33),
                     error_flags: wp.array(dtype=int), ambient_pressure: float, initial_porosity: float, density: wp.array(dtype=float),
                     elastic_modulus: float, time_step: float, grid_spacing: float, cfl_limit: float):
    particle_index = wp.tid()
    volume_ratio = wp.determinant(gradients[particle_index])
    porosity = porosities[particle_index]
    if porosity <= 0.0 or porosity >= 1.0 or not wp.isfinite(porosity) or volume_ratio <= 0.0:
        wp.atomic_max(error_flags, 0, 1)
        return
    pressure = float(gas_pressure(moles[particle_index], temperatures[particle_index], volume_ratio, porosity))
    # sigma_g = -N*R*T/(J*n) I; tau_g = J*sigma_g. Ambient pressure is external traction.
    total_stresses[particle_index] = solid_stresses[particle_index] + volume_ratio * (ambient_pressure - pressure) * wp.identity(n=3, dtype=float)
    # Conservative compressibility estimate, including possible plastic pore contraction.
    gas_bulk = pressure * (1.0 + (1.0 - initial_porosity + porosity) / porosity)
    wave_speed = wp.sqrt((elastic_modulus + gas_bulk) * volume_ratio / density[particle_index])
    if not wp.isfinite(pressure) or wave_speed * time_step / grid_spacing > cfl_limit:
        wp.atomic_max(error_flags, 2, 1)


@wp.kernel
def RefreshGas(old_moles: wp.array(dtype=wp.float64), new_moles: wp.array(dtype=wp.float64),
               gradients: wp.array(dtype=wp.mat33), porosities: wp.array(dtype=float),
               pressures: wp.array(dtype=wp.float64), temperatures: wp.array(dtype=float)):
    index = wp.tid()
    new_moles[index] = old_moles[index]
    if porosities[index] > 0.0 and porosities[index] < 1.0:
        pressures[index] = gas_pressure(old_moles[index], temperatures[index], wp.determinant(gradients[index]), porosities[index])
