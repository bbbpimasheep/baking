"""Run from the project root: python -m experiments.simulate_tet_mesh."""
import argparse
import json
import io
from contextlib import redirect_stdout, redirect_stderr
import warp as wp
from experiments.recording import SimulationRecorder
from dataclasses import replace
from pathlib import Path

from config import LoadExperimentConfig
from geometry.mesh_loader import CreateDemoMesh, LoadTetMesh
from geometry.mesh_validation import ValidateTetMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def ParseArguments(argument_values=None):
    argument_parser = argparse.ArgumentParser(
        description="3D Neo-Hookean MLS-MPM; values default to examples/oven_baking.txt",
        argument_default=argparse.SUPPRESS)
    argument_parser.add_argument("--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "oven_baking.txt",
        help="UTF-8 parameter TXT file; all physical settings are configured here")
    argument_parser.add_argument("--headless", action="store_true", help="simulate without Polyscope")
    argument_parser.add_argument("--gui", dest="headless", action="store_false", help="open Polyscope")
    argument_parser.add_argument("--steps", type=int, help="steps in headless mode")
    argument_parser.add_argument("--report-interval", type=int)
    argument_parser.add_argument("--log-path", type=Path, help="directory for results saved only after simulation finishes")
    argument_parser.add_argument("--output", type=Path, help="optional final NPZ snapshot")
    return argument_parser.parse_args(argument_values)


def BuildExperimentConfig(arguments):
    experiment_config = LoadExperimentConfig(arguments.config)
    run_options = {"headless": "headless", "steps": "steps",
                   "report_interval": "report_interval", "output": "output_path", "log_path": "log_path"}
    run_updates = {}
    for argument_name, option_name in run_options.items():
        if hasattr(arguments, argument_name):
            option_value = getattr(arguments, argument_name)
            if isinstance(option_value, Path):
                option_value = str(option_value.expanduser().resolve())
            run_updates[option_name] = option_value
    experiment_config = replace(experiment_config, run=replace(experiment_config.run, **run_updates))
    experiment_config.Validate()
    return experiment_config


def RunSimulation():
    arguments = ParseArguments()
    try:
        experiment_config = BuildExperimentConfig(arguments)
    except ValueError as error_value:
        raise SystemExit(str(error_value)) from error_value
    material_config = experiment_config.material
    simulation_config = experiment_config.simulation
    mesh_config = experiment_config.mesh
    sampling_config = experiment_config.sampling
    run_config = experiment_config.run

    if mesh_config.mesh_path:
        tet_mesh = LoadTetMesh(mesh_config.mesh_path, mesh_config.mesh_scale, mesh_config.mesh_translation)
    else:
        tet_mesh = CreateDemoMesh(mesh_config.demo_subdivisions)
        tet_mesh = ValidateTetMesh(tet_mesh.vertex_positions * mesh_config.mesh_scale + mesh_config.mesh_translation,
                                  tet_mesh.tet_indices)
    particle_samples = SampleParticles(
        tet_mesh, material_config.density, particles_per_tet=sampling_config.particles_per_tet,
        target_volume=simulation_config.grid_spacing ** 3 / sampling_config.target_particles_per_cell,
        random_seed=sampling_config.random_seed, max_particles=sampling_config.max_particles)

    simulator = MpmSimulator(tet_mesh, particle_samples, simulation_config,
                             material_config, experiment_config.initial.initial_velocity,
                             ding_config=experiment_config.ding,
                             transport_config=experiment_config.transport)
    result_recorder = SimulationRecorder(simulator, experiment_config)
    simulator.result_recorder = result_recorder
    RunSimulation.recorder = result_recorder
    if run_config.headless:
        while simulator.IsSettling or simulator.baking_steps < run_config.steps:
            # A rest may end on any step; do not overshoot the requested baking duration.
            step_count = 1 if simulator.IsSettling else min(run_config.report_interval, run_config.steps - simulator.baking_steps)
            simulator.Advance(step_count)
    else:
        from visualization.polyscope_viewer import PolyscopeViewer
        viewer = PolyscopeViewer(simulator)
        viewer.Show()
        if viewer.last_error:
            raise RuntimeError(viewer.last_error)
    if run_config.output_path:
        simulator.SaveSnapshot(run_config.output_path)


def Main():
    console_buffer = io.StringIO()
    error_value = None
    RunSimulation.recorder = None
    previous_log_level = wp.config.log_level
    wp.config.log_level = wp.LOG_ERROR + 1
    try:
        with redirect_stdout(console_buffer), redirect_stderr(console_buffer):
            RunSimulation()
    except (Exception, KeyboardInterrupt, SystemExit) as caught_error:
        error_value = caught_error
    finally:
        wp.config.log_level = previous_log_level
    result_recorder = RunSimulation.recorder
    if result_recorder is not None:
        status = 'completed' if error_value is None else 'failed'
        if isinstance(error_value, KeyboardInterrupt):
            status = 'interrupted'
        result_path = result_recorder.Finish(status, str(error_value) if error_value else '', console_buffer.getvalue())
        print(json.dumps(dict(status=status, results_path=str(result_path),
                              initial=result_recorder.initial_measurement,
                              final=result_recorder.last_measurement), indent=2))
    elif console_buffer.getvalue():
        print(console_buffer.getvalue(), end='')
    if error_value is not None:
        raise error_value


if __name__ == "__main__":
    Main()
