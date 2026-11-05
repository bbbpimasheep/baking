"""Ding 2019 solid skeleton, Sections 5.2-5.4 and 6.5.

Section 5.3.2: solved particle temperature changes eta and tau_y only.
Section 5.4: porosity = det(Fp) - 1 + initial_porosity. Solid-only mode uses
the old porosity; gas mode solves the end-state porosity consistently.
The paper does not specify this split or universal temperature table values.
Parameters freeze irreversibly at the fourth temperature knot; before that,
cooling retraces the piecewise linear curves. Thermal prestress c=0; no
prescribed expansion or fracture. Optional heat/water transport and saturation stress
are implemented separately in physics.thermal_water.
Finite eta > 0 uses the paper's squared positive-part penalty. eta has units
Pa^3 s for the unnormalized f in Pa^2, not ordinary fluid viscosity Pa s.
The local minimizer is solved by 32 Newton steps on a scalar overstress;
the paper does not specify that scalar algorithm. Its custom adjoint differentiates
the converged scalar equation, away from the yield switch. Spectral decomposition
uses float64 internally; stored state stays float32. Zero-porosity Fe volume is
normalized to its trial value to remove spectral reconstruction drift.
Gas mode adds Section 5.4 with gas content per reference volume and Kelvin temperature.
Gas-enabled plastic correction uses a numerically added end-porosity consistency solve;
legacy solid-only mode keeps its previous update. No global implicit solve.
The original main-paper f is retained, not the supplement's normalized penalty.

"""
from dataclasses import dataclass
import math
import numpy as np
import warp as wp
from physics.materials.elastic import ElasticMaterial
from physics.materials.gas import gas_constant, CombineGasStress, RefreshGas
from physics.materials.porous_return import solve_porosity, porosity_prediction


@dataclass
class DingState:
    """Gas content is mol / initial m^3; pressures are absolute Pa, not gauge pressure."""
    elastic_gradients: object
    plastic_gradients: object
    kirchhoff_stresses: object
    return_residuals: object
    porosities: object
    temperatures: object
    gelled: object
    yield_stresses: object
    viscosities: object
    gas_moles: object
    gas_pressures: object
    porosity_residuals: object
    reference_densities: object


def TemperatureFactor(temperature_value, temperature_knots, factor_values):
    """Section 5.3.2 / Figure 8, with constant endpoint extrapolation."""
    return float(np.interp(temperature_value, temperature_knots, factor_values))


@wp.func
def temperature_factor(temperature_value: float, temperature_knots: wp.vec4, factor_values: wp.vec4):
    factor_value = factor_values[0]
    for knot_index in range(3):
        if temperature_value >= temperature_knots[knot_index]:
            fraction_value = wp.min((temperature_value - temperature_knots[knot_index]) /
                (temperature_knots[knot_index + 1] - temperature_knots[knot_index]), 1.0)
            factor_value = ((1.0 - fraction_value) * factor_values[knot_index]
                            + fraction_value * factor_values[knot_index + 1])
    return factor_value


@wp.func
def porosity_coefficient(porosity: float):
    # Equation (12): (7/2)*n + 1, not 7/(2*n) + 1. n=0 is the von Mises limit.
    coefficient_value = float(0.0)
    if porosity > 0.0:
        coefficient_value = 9.0 / (4.0 * wp.pow(wp.log(1.0 / porosity), 2.0) * (3.5 * porosity + 1.0))
    return coefficient_value


@wp.func
def solve_overstress(q_squared: float, pressure_squared: float, dev_factor: float, mean_factor: float):
    root_value = float(0.0)
    if q_squared + pressure_squared > 1.0:
        for iteration_index in range(32):
            dev_denominator = 1.0 + dev_factor * root_value
            mean_denominator = 1.0 + mean_factor * root_value
            residual_value = root_value + 1.0 - q_squared / wp.pow(dev_denominator, 2.0) - pressure_squared / wp.pow(mean_denominator, 2.0)
            derivative_value = 1.0 + 2.0 * dev_factor * q_squared / wp.pow(dev_denominator, 3.0) + 2.0 * mean_factor * pressure_squared / wp.pow(mean_denominator, 3.0)
            root_value = root_value - residual_value / derivative_value
    return root_value


@wp.func_grad(solve_overstress)
def SolveOverstressGradient(q_squared: float, pressure_squared: float, dev_factor: float, mean_factor: float, adj_ret: float):
    if q_squared + pressure_squared > 1.0:
        root_value = solve_overstress(q_squared, pressure_squared, dev_factor, mean_factor)
        dev_denominator = 1.0 + dev_factor * root_value
        mean_denominator = 1.0 + mean_factor * root_value
        derivative_value = 1.0 + 2.0 * dev_factor * q_squared / wp.pow(dev_denominator, 3.0) + 2.0 * mean_factor * pressure_squared / wp.pow(mean_denominator, 3.0)
        wp.adjoint[q_squared] += adj_ret / (derivative_value * dev_denominator * dev_denominator)
        wp.adjoint[pressure_squared] += adj_ret / (derivative_value * mean_denominator * mean_denominator)
        wp.adjoint[dev_factor] -= adj_ret * 2.0 * q_squared * root_value / (derivative_value * wp.pow(dev_denominator, 3.0))
        wp.adjoint[mean_factor] -= adj_ret * 2.0 * pressure_squared * root_value / (derivative_value * wp.pow(mean_denominator, 3.0))


@wp.func
def ding_return(trial_gradient: wp.mat33, shear_modulus: float, lame_lambda: float,
                yield_stress: float, viscosity: float, pressure_coefficient: float, time_step: float):
    left_double, singular_double, right_double = wp.svd3(wp.mat33d(trial_gradient))
    left_vectors = wp.mat33(left_double)
    singular_values = wp.vec3(singular_double)
    right_vectors = wp.mat33(right_double)
    log_strain = wp.vec3(wp.log(singular_values[0]), wp.log(singular_values[1]), wp.log(singular_values[2]))
    trace_strain = log_strain[0] + log_strain[1] + log_strain[2]
    trial_stress = 2.0 * shear_modulus * log_strain + lame_lambda * trace_strain * wp.vec3(1.0)
    mean_stress = (trial_stress[0] + trial_stress[1] + trial_stress[2]) / 3.0
    deviator = trial_stress - mean_stress * wp.vec3(1.0)
    q_squared = 1.5 * wp.dot(deviator, deviator) / (yield_stress * yield_stress)
    pressure_squared = pressure_coefficient * mean_stress * mean_stress / (yield_stress * yield_stress)
    overstress = float(0.0)
    dev_factor = 6.0 * shear_modulus * time_step * yield_stress * yield_stress / viscosity
    mean_factor = (2.0 * pressure_coefficient / 3.0 * (2.0 * shear_modulus + 3.0 * lame_lambda)
                   * time_step * yield_stress * yield_stress / viscosity)
    overstress = solve_overstress(q_squared, pressure_squared, dev_factor, mean_factor)
    corrected_stress = (deviator / (1.0 + dev_factor * overstress)
                        + mean_stress / (1.0 + mean_factor * overstress) * wp.vec3(1.0))
    corrected_mean = (corrected_stress[0] + corrected_stress[1] + corrected_stress[2]) / 3.0
    corrected_strain = ((corrected_stress - corrected_mean * wp.vec3(1.0)) / (2.0 * shear_modulus)
                        + corrected_mean / (2.0 * shear_modulus + 3.0 * lame_lambda) * wp.vec3(1.0))
    elastic_gradient = left_vectors * wp.diag(wp.vec3(wp.exp(corrected_strain[0]),
                       wp.exp(corrected_strain[1]), wp.exp(corrected_strain[2]))) * wp.transpose(right_vectors)
    if pressure_coefficient == 0.0:
        # Exact zero-porosity flow is isochoric. Remove spectral reconstruction roundoff.
        elastic_gradient = elastic_gradient * wp.pow(wp.determinant(trial_gradient) / wp.determinant(elastic_gradient), 1.0 / 3.0)
    stress_matrix = left_vectors * wp.diag(corrected_stress) * wp.transpose(left_vectors)
    residual_norm = float(0.0)
    if q_squared + pressure_squared > 1.0:
        residual_norm = wp.abs(overstress + 1.0 - q_squared / wp.pow(1.0 + dev_factor * overstress, 2.0)
                              - pressure_squared / wp.pow(1.0 + mean_factor * overstress, 2.0))
    return elastic_gradient, stress_matrix, residual_norm


@wp.kernel
def UpdateDing(old_elastic: wp.array(dtype=wp.mat33), old_porosities: wp.array(dtype=float),
               old_gelled: wp.array(dtype=int), total_gradients: wp.array(dtype=wp.mat33),
               affine_velocities: wp.array(dtype=wp.mat33), new_elastic: wp.array(dtype=wp.mat33),
               new_plastic: wp.array(dtype=wp.mat33), new_stresses: wp.array(dtype=wp.mat33),
               residuals: wp.array(dtype=float), new_porosities: wp.array(dtype=float),
               temperatures: wp.array(dtype=float), new_gelled: wp.array(dtype=int),
               yield_stresses: wp.array(dtype=float), viscosities: wp.array(dtype=float),
               viscosity_parameter: wp.array(dtype=float), error_flags: wp.array(dtype=int),
               porosity_residuals: wp.array(dtype=float), coupled_porosity: int,
               shear_modulus: float, lame_lambda: float, yield_stress: float, initial_porosity: float,
               temperature_enabled: int, temperature_knots: wp.vec4,
               yield_factors: wp.vec4, viscosity_factors: wp.vec4, time_step: float,
               local_temperatures: wp.array(dtype=float)):
    particle_index = wp.tid()
    porosity = old_porosities[particle_index]
    if not wp.isfinite(porosity) or porosity < 0.0 or porosity >= 1.0:
        wp.atomic_max(error_flags, 0, 1)
        return
    gelled_value = old_gelled[particle_index]
    temperature_value = local_temperatures[particle_index]
    gelation_reached = int(temperature_value >= temperature_knots[3])
    effective_temperature = temperature_value
    yield_value = yield_stress
    viscosity_value = viscosity_parameter[0]
    if temperature_enabled != 0:
        if gelation_reached != 0:
            gelled_value = 1
        if gelled_value != 0:
            effective_temperature = temperature_knots[3]
        yield_value = yield_stress * temperature_factor(effective_temperature, temperature_knots, yield_factors)
        viscosity_value = viscosity_parameter[0] * temperature_factor(effective_temperature, temperature_knots, viscosity_factors)
    # Explicit MPM: old porosity defines this step's yield surface; corrected Fp defines the next.
    trial_gradient = (wp.identity(n=3, dtype=float) + time_step * affine_velocities[particle_index]) * old_elastic[particle_index]
    return_porosity = porosity
    consistency_residual = float(0.0)
    if coupled_porosity != 0:
        # Use the old porosity as the authoritative plastic-volume history; this
        # avoids accumulating float32 det(inv(Fe)*F) cancellation near closed pores.
        left_double, singular_double, right_double = wp.svd3(wp.mat33d(trial_gradient))
        strains = wp.vec3(wp.log(float(singular_double[0])), wp.log(float(singular_double[1])), wp.log(float(singular_double[2])))
        trial_stress = 2.0 * shear_modulus * strains + lame_lambda * (strains[0] + strains[1] + strains[2]) * wp.vec3(1.0)
        mean_value = (trial_stress[0] + trial_stress[1] + trial_stress[2]) / 3.0
        deviator = trial_stress - mean_value * wp.vec3(1.0)
        q_value = 1.5 * wp.dot(deviator, deviator) / (yield_value * yield_value)
        p_value = mean_value * mean_value / (yield_value * yield_value)
        dev_factor = 6.0 * shear_modulus * time_step * yield_value * yield_value / viscosity_value
        mean_scale = 2.0 / 3.0 * (2.0 * shear_modulus + 3.0 * lame_lambda) * time_step * yield_value * yield_value / viscosity_value
        exponent = 3.0 * mean_value / (2.0 * shear_modulus + 3.0 * lame_lambda)
        old_plastic_volume = wp.float64(1.0) - wp.float64(initial_porosity) + wp.float64(porosity)
        return_porosity = solve_porosity(q_value, p_value, dev_factor, mean_scale, exponent, old_plastic_volume, initial_porosity)
        predicted_porosity = porosity_prediction(wp.float64(return_porosity), q_value, p_value, dev_factor, mean_scale, exponent, old_plastic_volume, initial_porosity)
        consistency_residual = float(wp.abs(wp.float64(return_porosity) - predicted_porosity))
        if consistency_residual > 1.0e-6 or return_porosity <= 0.0 or return_porosity >= 1.0:
            wp.atomic_max(error_flags, 1, 1)
    elastic_gradient, stress_matrix, residual_value = ding_return(trial_gradient, shear_modulus,
        lame_lambda, yield_value, viscosity_value, porosity_coefficient(return_porosity), time_step)
    if coupled_porosity != 0:
        # Correct spectral roundoff while keeping total F and pore-volume history consistent.
        target_plastic_volume = wp.float64(1.0) - wp.float64(initial_porosity) + wp.float64(return_porosity)
        volume_scale = wp.determinant(wp.mat33d(total_gradients[particle_index])) / (target_plastic_volume * wp.determinant(wp.mat33d(elastic_gradient)))
        elastic_gradient = elastic_gradient * float(wp.pow(volume_scale, wp.float64(1.0 / 3.0)))
    plastic_gradient = wp.inverse(elastic_gradient) * total_gradients[particle_index]
    # Section 5.4. This is NOT det(Ftotal) and is NOT a gas-generation equation.
    new_porosity = wp.determinant(plastic_gradient) - 1.0 + initial_porosity
    if coupled_porosity != 0:
        new_porosity = return_porosity
    if initial_porosity == 0.0 and porosity == 0.0:
        # The exact nonporous flow has det(Fp)=1; discard only its float32 roundoff.
        new_porosity = 0.0
    if not wp.isfinite(new_porosity) or new_porosity < 0.0 or new_porosity >= 1.0:
        wp.atomic_max(error_flags, 0, 1)
    new_elastic[particle_index] = elastic_gradient
    new_plastic[particle_index] = plastic_gradient
    new_stresses[particle_index] = stress_matrix
    residuals[particle_index] = residual_value
    porosity_residuals[particle_index] = consistency_residual
    new_porosities[particle_index] = new_porosity
    temperatures[particle_index] = temperature_value
    new_gelled[particle_index] = gelled_value
    yield_stresses[particle_index] = yield_value
    viscosities[particle_index] = viscosity_value


class DingMaterial(ElasticMaterial):
    def __init__(self, material_config, ding_config, device_name, requires_grad=False, initial_temperature_k=293.15):
        super().__init__(material_config)
        ding_config.Validate()
        self.initial_temperature_k = initial_temperature_k
        self.ding_config = ding_config
        self.viscosity_parameter = wp.array([ding_config.viscosity], dtype=float, device=device_name,
                                            requires_grad=requires_grad)
        self.error_flags = wp.zeros(3, dtype=int, device=device_name)
        self.production_parameter = wp.array([ding_config.gas_production_rate], dtype=float, device=device_name, requires_grad=requires_grad)
        porosity = ding_config.porosity
        # Initial coefficient, also useful for single-step reference checks.
        self.pressure_coefficient = (0.0 if porosity == 0 else
            9.0 / (4.0 * math.log(1.0 / porosity) ** 2 * (3.5 * porosity + 1.0)))
        self.Reset()

    def Reset(self):
        self.error_flags.zero_()

    def InitializeState(self, particle_state):
        particle_count = len(particle_state.particle_positions)
        device_name = particle_state.particle_positions.device
        requires_grad = particle_state.particle_positions.requires_grad
        identity_values = np.tile(np.eye(3, dtype=np.float32), (particle_count, 1, 1))
        config = self.ding_config
        temperature_value = self.initial_temperature_k
        gelled_value = int(config.temperature_enabled and temperature_value >= config.temperature_knots_k[3])
        yield_factor = (TemperatureFactor(temperature_value, config.temperature_knots_k, config.yield_factors)
                        if config.temperature_enabled else 1.0)
        viscosity_factor = (TemperatureFactor(temperature_value, config.temperature_knots_k, config.viscosity_factors)
                            if config.temperature_enabled else 1.0)
        particle_state.material_state = DingState(
            elastic_gradients=wp.array(identity_values, dtype=wp.mat33, device=device_name, requires_grad=requires_grad),
            plastic_gradients=wp.array(identity_values, dtype=wp.mat33, device=device_name, requires_grad=requires_grad),
            kirchhoff_stresses=wp.zeros(particle_count, dtype=wp.mat33, device=device_name, requires_grad=requires_grad),
            return_residuals=wp.zeros(particle_count, dtype=float, device=device_name),
            porosities=wp.full(particle_count, config.porosity, dtype=float, device=device_name, requires_grad=requires_grad),
            temperatures=wp.full(particle_count, temperature_value, dtype=float, device=device_name, requires_grad=requires_grad),
            gelled=wp.full(particle_count, gelled_value, dtype=int, device=device_name),
            yield_stresses=wp.full(particle_count, config.yield_stress * yield_factor, dtype=float, device=device_name, requires_grad=requires_grad),
            viscosities=wp.full(particle_count, float(self.viscosity_parameter.numpy()[0]) * viscosity_factor, dtype=float, device=device_name, requires_grad=requires_grad),
            gas_moles=wp.full(particle_count, config.gas_initial_pressure_pa * config.porosity / (gas_constant * self.initial_temperature_k) if config.gas_enabled else 0.0,
                              dtype=wp.float64, device=device_name, requires_grad=requires_grad),
            gas_pressures=wp.full(particle_count, config.gas_initial_pressure_pa if config.gas_enabled else 0.0, dtype=wp.float64, device=device_name, requires_grad=requires_grad),
            porosity_residuals=wp.zeros(particle_count, dtype=float, device=device_name),
            reference_densities=wp.full(particle_count, self.material_config.density, dtype=float, device=device_name, requires_grad=requires_grad))

    def ComputeStress(self, particle_state, kirchhoff_stresses, error_flags):
        if not self.ding_config.gas_enabled:
            wp.copy(kirchhoff_stresses, particle_state.material_state.kirchhoff_stresses)
            return
        state = particle_state.material_state
        simulation = getattr(self, 'simulation_config', None)
        wp.launch(CombineGasStress, dim=len(kirchhoff_stresses), inputs=[state.kirchhoff_stresses,
            particle_state.deformation_gradients, state.porosities, state.gas_moles, state.temperatures,
            kirchhoff_stresses, self.error_flags, self.ding_config.gas_ambient_pressure_pa, self.ding_config.porosity,
            state.reference_densities, self.material_config.lame_lambda + 2 * self.material_config.shear_modulus,
            simulation.time_step if simulation else 0.0, simulation.grid_spacing if simulation else 1.0,
            simulation.cfl_limit if simulation else 0.4], device=kirchhoff_stresses.device)
        self.CheckState()

    def GasWaveSpeed(self, particle_state):
        internal = particle_state.material_state
        porosities = internal.porosities.numpy().astype(np.float64)
        pressures = internal.gas_pressures.numpy()
        volume_ratios = np.linalg.det(particle_state.deformation_gradients.numpy().astype(np.float64))
        gas_bulk = pressures * (1 + (1 - self.ding_config.porosity + porosities) / porosities)
        elastic_modulus = self.material_config.lame_lambda + 2 * self.material_config.shear_modulus
        return float(np.sqrt((elastic_modulus + gas_bulk) * volume_ratios / internal.reference_densities.numpy()).max())

    def CheckState(self):
        flags = self.error_flags.numpy()
        if flags[2]:
            raise RuntimeError("Gas acoustic CFL exceeds the explicit limit; reduce time_step. Pressure is not capped")
        if flags[1]:
            raise RuntimeError("Coupled porosity return has no resolved interior solution; reduce dt/loading. No pore-volume clipping or closed-pore law is applied")
        if flags[0]:
            raise RuntimeError("Ding porosity left [0, 1); no constitutive clamping is applied. Reduce dt or loading; the paper gives no closure beyond this domain")

    def AdvanceState(self, input_state, output_state, time_step):
        if output_state.material_state is None:
            self.InitializeState(output_state)
        old_material = input_state.material_state
        material_state = output_state.material_state
        config = self.ding_config
        wp.launch(UpdateDing, dim=len(output_state.particle_positions), inputs=[
            old_material.elastic_gradients, old_material.porosities, old_material.gelled,
            output_state.deformation_gradients, output_state.affine_velocities,
            material_state.elastic_gradients, material_state.plastic_gradients, material_state.kirchhoff_stresses,
            material_state.return_residuals, material_state.porosities, material_state.temperatures,
            material_state.gelled, material_state.yield_stresses, material_state.viscosities,
            self.viscosity_parameter, self.error_flags, material_state.porosity_residuals, int(config.gas_enabled),
            self.material_config.shear_modulus, self.material_config.lame_lambda,
            config.yield_stress, config.porosity, int(config.temperature_enabled),
            wp.vec4(*config.temperature_knots_k), wp.vec4(*config.yield_factors), wp.vec4(*config.viscosity_factors), time_step,
            old_material.temperatures],
            device=output_state.particle_positions.device)

        wp.copy(material_state.reference_densities, old_material.reference_densities)
        if config.gas_enabled:
            # Production already occurred in transport before the mechanical step.
            wp.launch(RefreshGas, len(output_state.particle_positions), inputs=[old_material.gas_moles,
                material_state.gas_moles, output_state.deformation_gradients, material_state.porosities,
                material_state.gas_pressures, material_state.temperatures], device=output_state.particle_positions.device)
