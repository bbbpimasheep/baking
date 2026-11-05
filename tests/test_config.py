from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from config import ExperimentConfig, LoadExperimentConfig
from experiments.simulate_tet_mesh import BuildExperimentConfig, Main, ParseArguments


def TestDefaultParameterFileMatchesDefaults(tmp_path):
    config_path = tmp_path / "defaults.txt"
    config_path.write_text("", encoding="utf-8")
    expected_config = ExperimentConfig()
    expected_config = replace(expected_config, run=replace(expected_config.run, log_path=str((config_path.parent / "outputs/logs").resolve())))
    assert LoadExperimentConfig(config_path) == expected_config


def TestConfigTypesCommentsAndRelativePaths(tmp_path):
    config_path = tmp_path / "parameters.txt"
    config_path.write_text(
        "# Parameter file comment\n[material]\ndensity = 1200 # kg/m^3\nyoung_modulus = 3e4\n"
        "[simulation]\ngravity = 1, -2, 3\ngrid_origin = -0.1, 0, 0.2\n"
        "[initial]\ninitial_velocity = 0.1, 0.2, 0.3\n"
        "[sampling]\nrandom_seed = 123\nparticles_per_tet = 4\n"
        "[mesh]\nmesh_path = \"models/input.mesh\"\n"
        "[run]\nheadless = true\nsteps = 20\noutput_path = outputs/final.npz\n",
        encoding="utf-8-sig")
    experiment_config = LoadExperimentConfig(config_path)
    assert experiment_config.material.density == 1200.0
    assert experiment_config.material.young_modulus == 30000.0
    assert experiment_config.simulation.gravity == (1.0, -2.0, 3.0)
    assert experiment_config.simulation.grid_origin == (-0.1, 0.0, 0.2)
    assert experiment_config.initial.initial_velocity == (0.1, 0.2, 0.3)
    assert experiment_config.sampling.random_seed == 123
    assert experiment_config.run.headless is True
    assert experiment_config.run.steps == 20
    assert experiment_config.mesh.mesh_path == str((tmp_path / "models/input.mesh").resolve())
    assert experiment_config.run.output_path == str((tmp_path / "outputs/final.npz").resolve())


@pytest.mark.parametrize("file_content,error_pattern", [
    ("[material]\nyoung_moduls=100\n", "unknown parameter.*young_moduls"),
    ("[unknown]\na=1\n", "unknown section"),
    ("[simulation]\ngravity=1,2\n", "gravity.*three finite"),
    ("[simulation]\ngravity=0,nan,0\n", "gravity.*three finite"),
    ("[run]\nheadless=maybe\n", "headless.*true or false"),
    ("[simulation]\ngrid_resolution=40.5\n", "grid_resolution"),
    ("[material]\ndensity=-1\n", "density must be positive"),
    ("[simulation]\ntime_step=1\n", "CFL"),
    ("[material]\ndensity=1000\ndensity=1200\n", "already exists"),
    ("[sampling]\nmax_particles=0\n", "max_particles must be positive"),
])
def TestInvalidConfigRejected(tmp_path, file_content, error_pattern):
    config_path = tmp_path / "invalid.txt"
    config_path.write_text(file_content, encoding="utf-8")
    with pytest.raises(ValueError, match=error_pattern):
        LoadExperimentConfig(config_path)


def TestMissingConfigRejected(tmp_path):
    with pytest.raises(ValueError, match="Configuration file"):
        LoadExperimentConfig(tmp_path / "missing.txt")


def TestCommandLineOnlyOverridesExplicitValues(tmp_path):
    config_path = tmp_path / "custom.txt"
    config_path.write_text(
        "[material]\ndensity=1300\n[simulation]\ngravity=1,2,3\n"
        "[run]\nheadless=true\nsteps=123\n", encoding="utf-8")
    arguments = ParseArguments(["--config", str(config_path)])
    experiment_config = BuildExperimentConfig(arguments)
    assert experiment_config.run.headless is True
    assert experiment_config.run.steps == 123
    assert experiment_config.material.density == 1300
    arguments = ParseArguments(["--config", str(config_path), "--gui", "--steps", "20",
                                 "--report-interval", "10", "--output", "custom.npz"])
    experiment_config = BuildExperimentConfig(arguments)
    assert experiment_config.run.headless is False
    assert experiment_config.material.density == 1300
    assert experiment_config.simulation.gravity == (1.0, 2.0, 3.0)
    assert experiment_config.run.steps == 20
    assert experiment_config.run.report_interval == 10
    assert experiment_config.run.output_path == str(Path("custom.npz").resolve())


@pytest.mark.parametrize("option_name", [
    "--mesh", "--scale", "--translate", "--device", "--grid-resolution",
    "--grid-origin", "--boundary-cells", "--cfl-limit", "--dx", "--dt", "--density",
    "--young", "--poisson", "--gravity", "--velocity", "--particles-per-tet",
    "--target-particles-per-cell", "--random-seed", "--max-particles",
    "--demo-subdivisions", "--substeps",
])
def TestPhysicalCommandLineOptionsRejected(option_name):
    with pytest.raises(SystemExit) as error_info:
        ParseArguments([option_name, "1"])
    assert error_info.value.code == 2


def TestConfigurationDrivesSimulationAndSnapshot(tmp_path, monkeypatch):
    config_path = tmp_path / "simulation.txt"
    config_path.write_text(
        "[material]\ndensity=2000\n[simulation]\ndevice_name=cpu\n"
        "gravity=0,0,0\ntime_step=0.0002\ngrid_origin=0.01,0.01,0.01\n"
        "[initial]\ninitial_velocity=0.4,0,0\n"
        "[sampling]\nparticles_per_tet=4\n"
        "[mesh]\ndemo_subdivisions=1\nmesh_scale=0.8\nmesh_translation=0.05,0.05,0.05\n"
        "[run]\nheadless=true\nsteps=3\nreport_interval=2\noutput_path=final.npz\n",
        encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--config", str(config_path)])
    Main()
    with np.load(tmp_path / "final.npz") as snapshot_data:
        assert len(snapshot_data["particle_positions"]) == 24
        assert snapshot_data["simulation_time"] == 3 * 0.0002
        np.testing.assert_allclose(snapshot_data["particle_masses"].sum(), 2000 * 0.027 * 0.8 ** 3)
        expected_vertices = snapshot_data["reference_vertices"].copy()
        expected_vertices[:, 0] += 0.4 * 3 * 0.0002
        np.testing.assert_allclose(snapshot_data["vertices"], expected_vertices, atol=1.0e-6)
