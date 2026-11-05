"""Simulation parameters use SI units: metre, kilogram, second, pascal."""

import configparser
from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class MaterialConfig:
    density: float = 1000.0
    young_modulus: float = 20000.0
    poisson_ratio: float = 0.3
    model: str = "neo_hookean"

    def Validate(self):
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("density must be positive and finite")
        if not math.isfinite(self.young_modulus) or self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive and finite")
        if self.model not in ("neo_hookean", "ding2019"):
            raise ValueError("material model must be neo_hookean or ding2019")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (-1, 0.5)")

    @property
    def shear_modulus(self):
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_lambda(self):
        return (self.young_modulus * self.poisson_ratio /
                ((1.0 + self.poisson_ratio) * (1.0 - 2.0 * self.poisson_ratio)))

    @property
    def wave_speed(self):
        return math.sqrt((self.lame_lambda + 2.0 * self.shear_modulus) / self.density)


@dataclass(frozen=True)
class SimulationConfig:
    grid_resolution: int = 40
    grid_spacing: float = 0.025
    grid_origin: tuple = (0.0, 0.0, 0.0)
    time_step: float = 1.0e-4
    gravity: tuple = (0.0, -9.81, 0.0)
    boundary_cells: int = 3
    steps_per_frame: int = 40
    device_name: str = "auto"
    cfl_limit: float = 0.4

    def Validate(self, material_config):
        material_config.Validate()
        if self.grid_resolution < 8 or self.boundary_cells < 2:
            raise ValueError("grid_resolution >= 8 and boundary_cells >= 2 are required")
        if self.boundary_cells * 2 + 2 >= self.grid_resolution:
            raise ValueError("boundary layers leave no usable interior")
        if not math.isfinite(self.grid_spacing) or self.grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive and finite")
        if not math.isfinite(self.time_step) or self.time_step <= 0.0:
            raise ValueError("time_step must be positive and finite")
        if self.steps_per_frame < 1:
            raise ValueError("steps_per_frame must be positive")
        if len(self.grid_origin) != 3 or len(self.gravity) != 3:
            raise ValueError("grid_origin and gravity must have three components")
        if not all(math.isfinite(value) for value in (*self.grid_origin, *self.gravity)):
            raise ValueError("grid_origin and gravity must be finite")
        # An initial elastic CFL check, not a guarantee under large deformation.
        if not math.isfinite(self.cfl_limit) or not 0.0 < self.cfl_limit <= 0.4:
            raise ValueError("cfl_limit must be in (0, 0.4]")
        if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device_name):
            raise ValueError("device_name must be auto, cpu, cuda, or cuda:<index>")
        if self.time_step * material_config.wave_speed / self.grid_spacing > self.cfl_limit:
            raise ValueError(f"initial elastic CFL > {self.cfl_limit}; reduce time_step or stiffness")


@dataclass(frozen=True)
class InitialConfig:
    initial_velocity: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SamplingConfig:
    particles_per_tet: int = 0
    target_particles_per_cell: int = 8
    random_seed: int = 42
    max_particles: int = 500000


@dataclass(frozen=True)
class MeshConfig:
    mesh_path: str = ""
    mesh_scale: float = 1.0
    mesh_translation: tuple = (0.0, 0.0, 0.0)
    demo_subdivisions: int = 6


@dataclass(frozen=True)
class RunConfig:
    headless: bool = False
    steps: int = 3000
    snapshot_interval: int = 0
    report_interval: int = 500
    output_path: str = ""
    log_path: str = "outputs/logs"


@dataclass(frozen=True)
class DingConfig:
    yield_stress: float = 2000.0
    viscosity: float = 1.0e10
    porosity: float = 0.0
    temperature_enabled: bool = False
    temperature_knots_k: tuple = (278.15, 293.15, 313.15, 353.15)
    yield_factors: tuple = (1.5, 1.0, 0.2, 2.0)
    viscosity_factors: tuple = (1.5, 1.0, 0.2, 0.05)
    gas_enabled: bool = False
    gas_initial_pressure_pa: float = 101325.0
    gas_ambient_pressure_pa: float = 101325.0
    gas_production_rate: float = 0.001
    gas_reaction_temperature_k: float = 333.15
    gas_reaction_width_k: float = 15.0

    def Validate(self):
        if (not all(math.isfinite(value) for value in (self.yield_stress, self.viscosity, self.porosity))
                or self.yield_stress <= 0 or self.viscosity <= 0 or not 0 <= self.porosity < 1):
            raise ValueError("Ding requires yield_stress > 0, viscosity > 0, and 0 <= porosity < 1")
        if (not all(math.isfinite(value) for value in (self.gas_initial_pressure_pa, self.gas_ambient_pressure_pa,
                self.gas_production_rate, self.gas_reaction_temperature_k, self.gas_reaction_width_k))
                or min(self.gas_initial_pressure_pa, self.gas_ambient_pressure_pa, self.gas_production_rate) < 0
                or min(self.gas_reaction_temperature_k, self.gas_reaction_width_k) <= 0):
            raise ValueError("Gas pressures/rate must be nonnegative; reaction temperatures/width must be positive and finite")
        if self.gas_enabled and self.porosity <= 0:
            raise ValueError("Gas mode requires positive initial porosity; this stage has no pore nucleation model")
        for field_name, value_list in (("temperature_knots_k", self.temperature_knots_k),
                                       ("yield_factors", self.yield_factors), ("viscosity_factors", self.viscosity_factors)):
            if not all(math.isfinite(value) and value > 0 for value in value_list):
                raise ValueError(f"Ding {field_name} must contain positive finite values")
        if any(len(value_list) != 4 for value_list in (self.temperature_knots_k, self.yield_factors, self.viscosity_factors)):
            raise ValueError("Ding temperature_knots_k, yield_factors and viscosity_factors require four entries")
        if any(right <= left for left, right in zip(self.temperature_knots_k, self.temperature_knots_k[1:])):
            raise ValueError("Ding temperature_knots_k must strictly increase")
        if not self.yield_factors[0] >= self.yield_factors[1] >= self.yield_factors[2] <= self.yield_factors[3]:
            raise ValueError("Ding yield_factors must decrease to the third knot then increase")


@dataclass(frozen=True)
class TransportConfig:
    settling_duration_s: float = 0.0
    settling_auto: bool = False
    settling_min_duration_s: float = 0.1
    settling_hold_duration_s: float = 0.1
    settling_speed_tolerance: float = 0.001
    settling_rate_tolerance: float = 0.001
    enabled: bool = False
    water_enabled: bool = True
    initial_temperature_k: float = 293.15
    oven_temperature_k: float = 453.15
    conductivity: float = 0.4
    specific_heat: float = 2800.0
    heat_transfer: float = 20.0
    tray_temperature_k: float = 453.15
    tray_heat_transfer: float = 100.0
    contact_distance: float = 0.005
    initial_saturation: float = 0.4
    water_density: float = 1000.0
    water_diffusivity: float = 1.0e-10
    surface_water_loss: float = 1.0e-7
    water_stress: float = 10000.0
    latent_heat: float = 0.0

    def Validate(self, material):
        if not math.isfinite(self.settling_duration_s) or self.settling_duration_s < 0:
            raise ValueError('settling_duration_s must be finite and nonnegative')
        if self.settling_duration_s > 0 and not self.enabled:
            raise ValueError('Settling requires transport.enabled=true for local constant-temperature initialization')
        if self.settling_auto:
            values = (self.settling_min_duration_s, self.settling_hold_duration_s,
                      self.settling_speed_tolerance, self.settling_rate_tolerance)
            if not all(math.isfinite(value) for value in values) or min(values) < 0 or min(values[1:]) <= 0:
                raise ValueError('Automatic settling requires nonnegative minimum time and positive hold time/tolerances')
            if self.settling_duration_s < self.settling_min_duration_s + self.settling_hold_duration_s:
                raise ValueError('Maximum settling duration must cover minimum plus hold duration')
        if not self.enabled:
            return
        if material.model != 'ding2019':
            raise ValueError('Transport requires the Ding material')
        if min(self.initial_temperature_k, self.oven_temperature_k, self.tray_temperature_k,
               self.specific_heat, self.water_density) <= 0:
            raise ValueError('Transport temperatures, heat capacity and water density must be positive')
        if min(self.conductivity, self.heat_transfer, self.tray_heat_transfer, self.contact_distance,
               self.water_diffusivity, self.surface_water_loss, self.water_stress, self.latent_heat) < 0:
            raise ValueError('Transport coefficients must be nonnegative')
        if not 0 <= self.initial_saturation < 1 or material.density <= self.water_density * self.initial_saturation:
            raise ValueError('Initial saturation must be in [0,1), leaving positive dry mass')


@dataclass(frozen=True)
class ExperimentConfig:
    material: MaterialConfig = field(default_factory=MaterialConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    initial: InitialConfig = field(default_factory=InitialConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    run: RunConfig = field(default_factory=RunConfig)
    ding: DingConfig = field(default_factory=DingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)

    def Validate(self):
        self.simulation.Validate(self.material)
        self.ding.Validate()
        self.transport.Validate(self.material)
        if not self.run.log_path.strip():
            raise ValueError("log_path must not be empty")
        for value_name, vector_value in (("initial_velocity", self.initial.initial_velocity),
                                          ("mesh_translation", self.mesh.mesh_translation)):
            if len(vector_value) != 3 or not all(math.isfinite(value) for value in vector_value):
                raise ValueError(f"{value_name} must have three finite components")
        if not math.isfinite(self.mesh.mesh_scale) or self.mesh.mesh_scale <= 0.0:
            raise ValueError("mesh_scale must be positive and finite")
        if self.mesh.demo_subdivisions < 1:
            raise ValueError("demo_subdivisions must be positive")
        if self.sampling.particles_per_tet < 0:
            raise ValueError("particles_per_tet must be nonnegative")
        if self.sampling.target_particles_per_cell < 1 or self.sampling.max_particles < 1:
            raise ValueError("target_particles_per_cell and max_particles must be positive")
        if self.sampling.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        if self.run.snapshot_interval < 0:
            raise ValueError("snapshot_interval must be nonnegative")
        if self.run.steps < 1 or self.run.report_interval < 1:
            raise ValueError("steps and report_interval must be positive")


def ParseConfigValue(raw_value, default_value, variable_length=False):
    if isinstance(default_value, bool):
        bool_values = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
        if raw_value.lower() not in bool_values:
            raise ValueError("expected true or false")
        return bool_values[raw_value.lower()]
    if isinstance(default_value, int):
        return int(raw_value)
    if isinstance(default_value, float):
        return float(raw_value)
    if isinstance(default_value, tuple):
        vector_values = tuple(float(value.strip()) for value in raw_value.split(","))
        if ((not variable_length and len(vector_values) != len(default_value))
                or not all(math.isfinite(value) for value in vector_values)):
            count_name = "three" if len(default_value) == 3 else str(len(default_value))
            raise ValueError(f"expected {count_name} finite numbers separated by commas")
        return vector_values
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "\"'":
        return raw_value[1:-1]
    return raw_value


def LoadExperimentConfig(config_path):
    """Read UTF-8 INI syntax from .txt, rejecting unknown keys and sections.

    Omitted settings use dataclass defaults. File paths are relative to this
    configuration file; CLI path overrides are relative to the working directory.
    """
    config_path = Path(config_path).expanduser().resolve()
    config_parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    config_parser.optionxform = str
    try:
        with config_path.open("r", encoding="utf-8-sig") as config_file:
            config_parser.read_file(config_file)
        if config_parser.defaults():
            raise ValueError("[DEFAULT] is unsupported; place values in their named sections")
        experiment_config = ExperimentConfig()
        section_defaults = {field_info.name: getattr(experiment_config, field_info.name)
                            for field_info in fields(experiment_config)}
        for section_name in config_parser.sections():
            if section_name not in section_defaults:
                raise ValueError(f"unknown section [{section_name}]")
        section_configs = {}
        for section_name, default_config in section_defaults.items():
            default_values = {field_info.name: getattr(default_config, field_info.name)
                              for field_info in fields(default_config)}
            parsed_values = {}
            if config_parser.has_section(section_name):
                for option_name, raw_value in config_parser.items(section_name):
                    if option_name not in default_values:
                        raise ValueError(f"unknown parameter [{section_name}] {option_name}")
                    try:
                        parsed_values[option_name] = ParseConfigValue(raw_value, default_values[option_name])
                    except ValueError as error_value:
                        raise ValueError(f"[{section_name}] {option_name}: {error_value}") from error_value
            if section_name == "run":
                parsed_values.setdefault("log_path", default_config.log_path)
            for option_name in ("mesh_path", "output_path", "log_path"):
                if parsed_values.get(option_name):
                    file_path = Path(parsed_values[option_name]).expanduser()
                    if not file_path.is_absolute():
                        file_path = config_path.parent / file_path
                    parsed_values[option_name] = str(file_path.resolve())
            section_configs[section_name] = type(default_config)(**parsed_values)
        experiment_config = ExperimentConfig(**section_configs)
        experiment_config.Validate()
        return experiment_config
    except (OSError, UnicodeError, ValueError, configparser.Error) as error_value:
        raise ValueError(f"Configuration file {config_path}: {error_value}") from error_value
