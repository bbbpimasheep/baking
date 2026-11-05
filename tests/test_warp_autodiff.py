from dataclasses import replace
import numpy as np
import warp as wp
from config import MaterialConfig, SimulationConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


@wp.kernel
def SetInitialVelocity(velocity_parameter: wp.array(dtype=float),
                       particle_velocities: wp.array(dtype=wp.vec3)):
    particle_index = wp.tid()
    particle_velocities[particle_index] = wp.vec3(velocity_parameter[0], 0.0, 0.0)


@wp.kernel
def ComputeDisplacementLoss(final_positions: wp.array(dtype=wp.vec3),
                             initial_positions: wp.array(dtype=wp.vec3),
                             loss_value: wp.array(dtype=float), inverse_count: float):
    particle_index = wp.tid()
    wp.atomic_add(loss_value, 0, inverse_count *
                  (final_positions[particle_index][0] - initial_positions[particle_index][0]))


def EvaluateShortTrajectory(velocity_value, compute_gradient):
    tet_mesh = CreateDemoMesh(1)
    particle_samples = SampleParticles(tet_mesh, 1000.0, 4)
    simulator = MpmSimulator(tet_mesh, particle_samples,
                             replace(SimulationConfig(), grid_resolution=20, grid_spacing=0.05,
                                     time_step=0.001, gravity=(0.0, 0.0, 0.0), device_name="cpu"),
                             MaterialConfig(), retain_history=True)
    particle_count = simulator.particle_data.particle_count
    initial_positions = simulator.current_state.particle_positions
    velocity_parameter = wp.array([velocity_value], dtype=float, device="cpu", requires_grad=True)
    loss_value = wp.zeros(1, dtype=float, device="cpu", requires_grad=True)
    tape_context = wp.Tape()
    with tape_context:
        wp.launch(SetInitialVelocity, dim=particle_count,
                  inputs=[velocity_parameter, simulator.current_state.particle_velocities], device="cpu")
        for step_index in range(3):
            simulator.Step()
        wp.launch(ComputeDisplacementLoss, dim=particle_count, inputs=[
            simulator.current_state.particle_positions, initial_positions, loss_value,
            1.0 / particle_count], device="cpu")
    simulator.CheckState()
    gradient_value = None
    if compute_gradient:
        tape_context.backward(loss_value)
        gradient_value = float(velocity_parameter.grad.numpy()[0])
    return float(loss_value.numpy()[0]), gradient_value


def TestMpmVelocityGradientAgainstFiniteDifference():
    loss_value, gradient_value = EvaluateShortTrajectory(0.2, True)
    epsilon_value = 0.1
    positive_loss, _ = EvaluateShortTrajectory(0.2 + epsilon_value, False)
    negative_loss, _ = EvaluateShortTrajectory(0.2 - epsilon_value, False)
    finite_difference = (positive_loss - negative_loss) / (2.0 * epsilon_value)
    np.testing.assert_allclose(gradient_value, finite_difference, rtol=0.01, atol=1.0e-6)
    np.testing.assert_allclose(gradient_value, 0.003, rtol=0.01, atol=1.0e-6)
