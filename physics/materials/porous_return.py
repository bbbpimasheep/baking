"""Numerical addition: end-state porosity consistency for the Ding return map.

The main-paper yield function is retained. Porosity is solved by bounded
bisection, not clipped after plastic flow. Its custom adjoint differentiates
the converged consistency equation, not the bisection decisions.
No new pore-nucleation or closed-pore constitutive law is introduced.
"""
import warp as wp


@wp.func
def pressure_coefficient64(porosity: wp.float64):
    coefficient = wp.float64(0.0)
    if porosity > wp.float64(0.0):
        coefficient = wp.float64(9.0) / (wp.float64(4.0) * wp.log(wp.float64(1.0) / porosity) ** wp.float64(2.0) * (wp.float64(3.5) * porosity + wp.float64(1.0)))
    return coefficient


@wp.func
def overstress64(q_value: wp.float64, p_value: wp.float64, dev_factor: wp.float64, mean_factor: wp.float64):
    root_value = wp.float64(0.0)
    if q_value + p_value > wp.float64(1.0):
        for iteration_index in range(48):
            u_value = wp.float64(1.0) + dev_factor * root_value
            v_value = wp.float64(1.0) + mean_factor * root_value
            residual = root_value + wp.float64(1.0) - q_value / (u_value * u_value) - p_value / (v_value * v_value)
            derivative = wp.float64(1.0) + wp.float64(2.0) * dev_factor * q_value / (u_value ** wp.float64(3.0)) + wp.float64(2.0) * mean_factor * p_value / (v_value ** wp.float64(3.0))
            root_value -= residual / derivative
            if wp.abs(residual) < wp.float64(1.0e-12):
                break
    return root_value


@wp.func
def porosity_prediction(porosity: wp.float64, q_value: float, p_value: float, dev_factor: float,
                        mean_scale: float, exponent: float, plastic_volume: wp.float64, initial_porosity: float):
    coefficient = pressure_coefficient64(porosity)
    root_value = overstress64(wp.float64(q_value), coefficient * wp.float64(p_value), wp.float64(dev_factor), wp.float64(mean_scale) * coefficient)
    denominator = wp.float64(1.0) + wp.float64(mean_scale) * coefficient * root_value
    return plastic_volume * wp.exp(wp.float64(exponent) * (wp.float64(1.0) - wp.float64(1.0) / denominator)) - wp.float64(1.0) + wp.float64(initial_porosity)


@wp.func
def solve_porosity(q_value: float, p_value: float, dev_factor: float, mean_scale: float,
                   exponent: float, plastic_volume: wp.float64, initial_porosity: float):
    trial_porosity = plastic_volume - wp.float64(1.0) + wp.float64(initial_porosity)
    if trial_porosity > wp.float64(0.0) and trial_porosity < wp.float64(1.0):
        trial_coefficient = pressure_coefficient64(trial_porosity)
        if wp.float64(q_value) + trial_coefficient * wp.float64(p_value) <= wp.float64(1.0):
            return float(trial_porosity)
    lower_value = wp.float64(0.0)
    upper_value = wp.float64(1.0 - 1.0e-9)
    # Endpoints are mathematical limits, not an artificial positive porosity floor.
    for iteration_index in range(64):
        middle_value = (lower_value + upper_value) * wp.float64(0.5)
        predicted_value = porosity_prediction(middle_value, q_value, p_value, dev_factor, mean_scale, exponent, plastic_volume, initial_porosity)
        if middle_value < predicted_value:
            lower_value = middle_value
        else:
            upper_value = middle_value
    return float((lower_value + upper_value) * wp.float64(0.5))


@wp.func_grad(solve_porosity)
def SolvePorosityGradient(q_value: float, p_value: float, dev_factor: float, mean_scale: float,
                          exponent: float, plastic_volume: wp.float64, initial_porosity: float, adj_ret: float):
    porosity = wp.float64(solve_porosity(q_value, p_value, dev_factor, mean_scale, exponent, plastic_volume, initial_porosity))
    a_value = pressure_coefficient64(porosity)
    a_derivative = a_value * (wp.float64(2.0) / (porosity * wp.log(wp.float64(1.0) / porosity)) - wp.float64(3.5) / (wp.float64(3.5) * porosity + wp.float64(1.0)))
    q_double = wp.float64(q_value)
    p_double = wp.float64(p_value)
    d_double = wp.float64(dev_factor)
    h_double = wp.float64(mean_scale)
    root_value = overstress64(q_double, a_value * p_double, d_double, h_double * a_value)
    u_value = wp.float64(1.0) + d_double * root_value
    v_value = wp.float64(1.0) + h_double * a_value * root_value
    root_derivative = wp.float64(1.0) + wp.float64(2.0) * d_double * q_double / (u_value ** wp.float64(3.0)) + wp.float64(2.0) * h_double * a_value * a_value * p_double / (v_value ** wp.float64(3.0))
    r_q = wp.float64(0.0)
    r_p = wp.float64(0.0)
    r_d = wp.float64(0.0)
    r_h = wp.float64(0.0)
    r_a = wp.float64(0.0)
    if q_double + a_value * p_double > wp.float64(1.0):
        r_q = wp.float64(1.0) / (root_derivative * u_value * u_value)
        r_p = a_value / (root_derivative * v_value * v_value)
        r_d = -wp.float64(2.0) * q_double * root_value / (root_derivative * u_value ** wp.float64(3.0))
        r_h = -wp.float64(2.0) * a_value * a_value * p_double * root_value / (root_derivative * v_value ** wp.float64(3.0))
        r_a = p_double * (wp.float64(1.0) - h_double * a_value * root_value) / (root_derivative * v_value ** wp.float64(3.0))
    exponential = wp.exp(wp.float64(exponent) * (wp.float64(1.0) - wp.float64(1.0) / v_value))
    new_volume = plastic_volume * exponential
    prefactor = new_volume * wp.float64(exponent) / (v_value * v_value)
    residual_derivative = wp.float64(1.0) - prefactor * h_double * a_derivative * (root_value + a_value * r_a)
    adjoint_value = wp.float64(adj_ret) / residual_derivative
    wp.adjoint[q_value] += float(adjoint_value * prefactor * h_double * a_value * r_q)
    wp.adjoint[p_value] += float(adjoint_value * prefactor * h_double * a_value * r_p)
    wp.adjoint[dev_factor] += float(adjoint_value * prefactor * h_double * a_value * r_d)
    wp.adjoint[mean_scale] += float(adjoint_value * prefactor * (a_value * root_value + h_double * a_value * r_h))
    wp.adjoint[exponent] += float(adjoint_value * new_volume * (wp.float64(1.0) - wp.float64(1.0) / v_value))
    wp.adjoint[plastic_volume] += adjoint_value * exponential
    wp.adjoint[initial_porosity] += float(adjoint_value)
