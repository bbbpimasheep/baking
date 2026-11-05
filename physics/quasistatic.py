"""Read-only kinematic stopping diagnostics; no forces or state corrections.

Maximum particle speed and symmetric affine velocity gradient detect translation
and deformation. Relative plastic-gradient rate detects local ongoing relaxation.
These are tolerances for a numerical quasi-static state, not a force-equilibrium
proof. The discrete stopping event is not differentiated by the adjoint.
"""
import warp as wp


@wp.kernel
def SettlingMetrics(velocities: wp.array(dtype=wp.vec3), affine: wp.array(dtype=wp.mat33),
                    old_plastic: wp.array(dtype=wp.mat33), new_plastic: wp.array(dtype=wp.mat33),
                    metrics: wp.array(dtype=float), time_step: float):
    index = wp.tid()
    symmetric = 0.5 * (affine[index] + wp.transpose(affine[index]))
    plastic_change = new_plastic[index] - old_plastic[index]
    plastic_norm = wp.sqrt(wp.ddot(old_plastic[index], old_plastic[index]))
    wp.atomic_max(metrics, 0, wp.length(velocities[index]))
    wp.atomic_max(metrics, 1, wp.sqrt(wp.ddot(symmetric, symmetric)))
    wp.atomic_max(metrics, 2, wp.sqrt(wp.ddot(plastic_change, plastic_change)) / (time_step * plastic_norm))
