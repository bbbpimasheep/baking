from dataclasses import replace
import numpy as np
import warp as wp
from config import MaterialConfig, SimulationConfig
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def TestElasticReferenceRotationAndStretch():
    tet_mesh = CreateDemoMesh(1)
    material_config = MaterialConfig()
    simulator = MpmSimulator(tet_mesh, SampleParticles(tet_mesh, 1000.0, 1),
                             replace(SimulationConfig(), device_name="cpu"), material_config)
    deformation_gradients = np.tile(np.eye(3, dtype=np.float32), (6, 1, 1))
    rotation_angle = 0.4
    deformation_gradients[1] = [[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                               [np.sin(rotation_angle), np.cos(rotation_angle), 0], [0, 0, 1]]
    deformation_gradients[2] = np.diag([1.1, 1.0, 1.0])
    deformation_gradients[3] = np.diag([0.9, 1.0, 1.0])
    simulator.current_state.deformation_gradients.assign(deformation_gradients)
    simulator.material.ComputeStress(simulator.current_state, simulator.kirchhoff_stresses,
                                      simulator.error_flags)
    actual_stresses = simulator.kirchhoff_stresses.numpy()
    expected_stresses = []
    for deformation_gradient in deformation_gradients:
        expected_stresses.append(material_config.shear_modulus *
                                 (deformation_gradient @ deformation_gradient.T - np.eye(3)) +
                                 material_config.lame_lambda * np.log(np.linalg.det(deformation_gradient))
                                 * np.eye(3))
    np.testing.assert_allclose(actual_stresses, expected_stresses, rtol=1.0e-5, atol=0.005)
    np.testing.assert_allclose(actual_stresses[:2], 0.0, atol=0.005)
    assert actual_stresses[2, 0, 0] > 0.0
    assert actual_stresses[3, 0, 0] < 0.0
    deformation_gradients[0, 0, 0] = -1.0
    simulator.current_state.deformation_gradients.assign(deformation_gradients)
    simulator.material.ComputeStress(simulator.current_state, simulator.kirchhoff_stresses,
                                      simulator.error_flags)
    assert simulator.error_flags.numpy()[0] == 1
