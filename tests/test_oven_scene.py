from dataclasses import replace
from pathlib import Path
import numpy as np
from config import LoadExperimentConfig
from experiments.simulate_tet_mesh import ParseArguments, BuildExperimentConfig
from geometry.mesh_loader import LoadTetMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def BuildOvenScene(config_name='examples/oven_baking.txt', gravity=None):
    root = Path(__file__).resolve().parents[1]
    config = LoadExperimentConfig(root / config_name)
    simulation = replace(config.simulation, device_name='cpu')
    if gravity is not None:
        simulation = replace(simulation, gravity=gravity)
    mesh = LoadTetMesh(config.mesh.mesh_path, config.mesh.mesh_scale, config.mesh.mesh_translation)
    samples = SampleParticles(mesh, config.material.density, particles_per_tet=4)
    simulator = MpmSimulator(mesh, samples, simulation, config.material,
        initial_velocity=config.initial.initial_velocity, ding_config=config.ding,
        transport_config=config.transport)
    return config, simulator


def TestOvenStartsNearFloorAndFallsWithTransportEnabled():
    config, simulator = BuildOvenScene()
    _, control = BuildOvenScene(gravity=(0., 0., 0.))
    floor = config.simulation.grid_origin[1] + (config.simulation.boundary_cells - 1) * config.simulation.grid_spacing
    initial_bottom = simulator.tet_mesh.vertex_positions[:, 1].min()
    assert 0.0 < initial_bottom - floor <= .001 + 1e-12
    assert config.simulation.gravity == (0., -9.81, 0.)
    assert simulator.transport is not None and config.ding.gas_enabled
    masses = simulator.Masses().numpy().astype(np.float64)
    initial_center_y = np.average(simulator.current_state.particle_positions.numpy()[:, 1], weights=masses)
    simulator.Advance(40)
    control.Advance(40)
    state = simulator.current_state
    masses = simulator.Masses().numpy().astype(np.float64)
    center_y = np.average(state.particle_positions.numpy()[:, 1], weights=masses)
    mean_velocity_y = np.average(state.particle_velocities.numpy()[:, 1], weights=masses)
    control_velocity_y = np.average(control.current_state.particle_velocities.numpy()[:, 1], weights=control.Masses().numpy())
    assert center_y < initial_center_y - 1e-6
    assert mean_velocity_y < -.005
    assert mean_velocity_y < control_velocity_y - .005
    if simulator.step_count <= simulator.settling_steps:
        np.testing.assert_allclose(state.transport_state.temperatures.numpy(), config.transport.initial_temperature_k, atol=1e-12)
    else:
        assert state.transport_state.temperatures.numpy().max() > config.transport.initial_temperature_k


def TestDefaultSceneLoadsAtGround():
    scene = BuildExperimentConfig(ParseArguments([]))
    mesh = LoadTetMesh(scene.mesh.mesh_path, scene.mesh.mesh_scale, scene.mesh.mesh_translation)
    floor = scene.simulation.grid_origin[1] + (scene.simulation.boundary_cells - 1) * scene.simulation.grid_spacing
    assert 0.0 < mesh.vertex_positions[:, 1].min() - floor <= .001 + 1e-12
    assert scene.simulation.gravity[1] == -9.81
