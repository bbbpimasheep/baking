import warp as wp


@wp.func
def neo_hookean_kirchhoff(deformation_gradient: wp.mat33, shear_modulus: float,
                         lame_lambda: float):
    volume_ratio = wp.determinant(deformation_gradient)
    identity_matrix = wp.identity(n=3, dtype=float)
    return (shear_modulus * (deformation_gradient * wp.transpose(deformation_gradient) - identity_matrix)
            + lame_lambda * wp.log(volume_ratio) * identity_matrix)


@wp.kernel
def ComputeElasticStress(deformation_gradients: wp.array(dtype=wp.mat33),
                         kirchhoff_stresses: wp.array(dtype=wp.mat33),
                         shear_modulus: float, lame_lambda: float,
                         error_flags: wp.array(dtype=int)):
    particle_index = wp.tid()
    deformation_gradient = deformation_gradients[particle_index]
    volume_ratio = wp.determinant(deformation_gradient)
    if not wp.isfinite(volume_ratio) or volume_ratio <= 0.0:
        wp.atomic_max(error_flags, 0, 1)
        kirchhoff_stresses[particle_index] = wp.mat33(0.0)
    else:
        kirchhoff_stresses[particle_index] = neo_hookean_kirchhoff(
            deformation_gradient, shear_modulus, lame_lambda)


class ElasticMaterial:
    """Compressible Neo-Hookean: psi=mu/2*(I1-3)-mu*log(J)+lambda/2*log(J)^2.

    Transfer kernels consume Kirchhoff stress tau=P*F^T and reference volume.
    Future models own state.material_state and AdvanceState(), not P2G/G2P.
    """

    def __init__(self, material_config):
        material_config.Validate()
        self.material_config = material_config

    def ComputeStress(self, particle_state, kirchhoff_stresses, error_flags):
        wp.launch(ComputeElasticStress, dim=len(kirchhoff_stresses), inputs=[
            particle_state.deformation_gradients, kirchhoff_stresses,
            self.material_config.shear_modulus, self.material_config.lame_lambda,
            error_flags], device=kirchhoff_stresses.device)

    def AdvanceState(self, input_state, output_state, time_step):
        # Elasticity has no history variables. A viscous/plastic model must
        # allocate and update distinct internal-state arrays in history mode.
        output_state.material_state = None
