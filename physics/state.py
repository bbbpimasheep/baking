from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass
class ParticleState:
    particle_positions: object
    particle_velocities: object
    deformation_gradients: object
    affine_velocities: object
    material_state: object = None
    transport_state: object = None


@dataclass
class ParticleData:
    particle_masses: object
    reference_volumes: object
    particle_count: int


def AllocateState(particle_count, device_name, requires_grad=False):
    return ParticleState(
        wp.zeros(particle_count, dtype=wp.vec3, device=device_name, requires_grad=requires_grad),
        wp.zeros(particle_count, dtype=wp.vec3, device=device_name, requires_grad=requires_grad),
        wp.zeros(particle_count, dtype=wp.mat33, device=device_name, requires_grad=requires_grad),
        wp.zeros(particle_count, dtype=wp.mat33, device=device_name, requires_grad=requires_grad))


def CreateParticleState(particle_samples, device_name, initial_velocity=(0.0, 0.0, 0.0),
                        requires_grad=False):
    particle_count = len(particle_samples.particle_positions)
    initial_velocity = np.asarray(initial_velocity, dtype=np.float32)
    if initial_velocity.shape != (3,) or not np.isfinite(initial_velocity).all():
        raise ValueError("initial_velocity must have three finite components")
    particle_state = AllocateState(particle_count, device_name, requires_grad)
    particle_state.particle_positions.assign(
        particle_samples.particle_positions.astype(np.float32))
    particle_state.particle_velocities.assign(np.tile(initial_velocity, (particle_count, 1)))
    particle_state.deformation_gradients.assign(
        np.tile(np.eye(3, dtype=np.float32), (particle_count, 1, 1)))
    particle_data = ParticleData(
        wp.array(particle_samples.particle_masses, dtype=float, device=device_name),
        wp.array(particle_samples.reference_volumes, dtype=float, device=device_name), particle_count)
    return particle_state, particle_data
