from dataclasses import replace
import csv
import json
import numpy as np
from config import ExperimentConfig, MaterialConfig, SimulationConfig
from experiments.recording import SimulationRecorder, ExtractSurface
from geometry.mesh_loader import CreateDemoMesh
from geometry.particle_sampling import SampleParticles
from physics.mpm import MpmSimulator


def BuildRecordedSimulator():
    tet_mesh = CreateDemoMesh(2)
    return MpmSimulator(tet_mesh, SampleParticles(tet_mesh, 1000.0, 4),
        replace(SimulationConfig(), device_name='cpu', gravity=(0.0, 0.0, 0.0)),
        MaterialConfig(), initial_velocity=(0.1, -0.2, 0.05))


def TestRecordingDoesNotChangeSimulationAndWritesOnlyAtFinish(tmp_path, capsys):
    baseline_simulator = BuildRecordedSimulator()
    recorded_simulator = BuildRecordedSimulator()
    experiment_config = ExperimentConfig()
    experiment_config = replace(experiment_config, run=replace(experiment_config.run, log_path=str(tmp_path / 'logs')))
    recorder = SimulationRecorder(recorded_simulator, experiment_config)
    recorded_simulator.result_recorder = recorder
    capsys.readouterr()
    baseline_simulator.Advance(20)
    recorded_simulator.Advance(20)
    assert not (tmp_path / 'logs').exists()
    assert capsys.readouterr().out == ''
    for state_name in ('particle_positions', 'particle_velocities', 'deformation_gradients', 'affine_velocities'):
        np.testing.assert_array_equal(getattr(baseline_simulator.current_state, state_name).numpy(),
                                      getattr(recorded_simulator.current_state, state_name).numpy())
    assert len(recorder.height_history) == 21
    np.testing.assert_allclose(np.asarray(recorder.height_history)[:, 1], 0.3, atol=2e-6)
    recorder.QueueSnapshot()
    assert not (tmp_path / 'logs').exists()
    run_path = recorder.Finish()
    assert recorder.Finish() == run_path
    summary_data = json.loads((run_path / 'summary.json').read_text())
    np.testing.assert_allclose(summary_data['initial']['particle_volume_m3'], 0.027)
    np.testing.assert_allclose(summary_data['final']['particle_volume_m3'], 0.027, rtol=1e-5)
    np.testing.assert_allclose(summary_data['final']['mesh_volume_m3'], 0.027, rtol=1e-5)
    assert summary_data['final']['inverted_tets'] == 0
    assert (run_path / 'final_surface.obj').exists()
    assert (run_path / 'height_history.svg').exists()
    with (run_path / 'height_history.csv').open() as history_file:
        rows = list(csv.DictReader(history_file))
    assert len(rows) == 21
    assert float(rows[-1]['time_s']) == 20 * recorded_simulator.simulation_config.time_step
    surface_vertices, surface_faces = ExtractSurface(recorded_simulator.vertex_positions.numpy(),
                                                     recorded_simulator.tet_mesh.tet_indices)
    centered_vertices = surface_vertices.astype(float) - surface_vertices.mean(axis=0)
    triangles = centered_vertices[surface_faces]
    surface_volume = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    np.testing.assert_allclose(surface_volume, 0.027, rtol=1e-5)
    recorded_simulator.Reset()
    assert len(recorder.height_history) == 1
    assert len(recorder.pending_snapshots) == 0


def TestAffineDimensionsAndVolumes(tmp_path):
    simulator = BuildRecordedSimulator()
    experiment_config = ExperimentConfig()
    experiment_config = replace(experiment_config, run=replace(experiment_config.run, log_path=str(tmp_path)))
    recorder = SimulationRecorder(simulator, experiment_config)
    affine_matrix = np.diag([1.2, 0.8, 1.1])
    simulator.current_state.particle_positions.assign(
        (simulator.particle_samples.particle_positions @ affine_matrix.T).astype(np.float32))
    simulator.current_state.deformation_gradients.assign(
        np.tile(affine_matrix.astype(np.float32), (simulator.particle_data.particle_count, 1, 1)))
    simulator.step_count = 1
    recorder.Record()
    final_values = recorder.last_measurement
    np.testing.assert_allclose([final_values['length_x_m'], final_values['width_z_m'], final_values['height_y_m']],
                               [0.36, 0.33, 0.24], atol=1e-6)
    np.testing.assert_allclose(final_values['particle_volume_m3'], 0.027 * np.linalg.det(affine_matrix), rtol=1e-6)
    np.testing.assert_allclose(final_values['mesh_volume_m3'], final_values['particle_volume_m3'], rtol=1e-6)


def TestMainDefersAllOutputAndFilesUntilExit(tmp_path, monkeypatch, capsys):
    import sys
    from experiments.simulate_tet_mesh import Main
    config_path = tmp_path / 'run.txt'
    config_path.write_text('[simulation]\ndevice_name=cpu\n[mesh]\ndemo_subdivisions=1\n'
                           '[sampling]\nparticles_per_tet=4\n[run]\nheadless=true\n'
                           'steps=4\nreport_interval=2\nlog_path=logs\n', encoding='utf-8')
    original_step = MpmSimulator.Step
    observed_steps = []
    def CheckedStep(simulator):
        assert not (tmp_path / 'logs').exists()
        assert sys.stdout.getvalue() == ''
        original_step(simulator)
        observed_steps.append(simulator.step_count)
    monkeypatch.setattr(MpmSimulator, 'Step', CheckedStep)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--config', str(config_path)])
    Main()
    assert observed_steps == [1, 2, 3, 4]
    output_data = json.loads(capsys.readouterr().out)
    assert output_data['status'] == 'completed'
    assert output_data['final']['step'] == 4
    run_directories = list((tmp_path / 'logs').iterdir())
    assert len(run_directories) == 1
    assert (run_directories[0] / 'summary.json').exists()
