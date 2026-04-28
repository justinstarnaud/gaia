"""Pydantic configuration models and CA-calibrated defaults for DamForge.

All parameters, enums, and physical reference values live in this module.
No magic numbers appear anywhere else in the codebase.

Coordinate system
-----------------
  x : horizontal, x=0 at dam centre-line (m)
  y : vertical,   y=0 at crest, negative downward (m)
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DamType(str, Enum):
    """Embankment dam archetype."""

    HOMOGENEOUS = "homogeneous"
    ZONED = "zoned"
    PUDDLE_CORE = "puddle_core"


class ScenarioType(str, Enum):
    """Anomaly composition for a scenario."""

    CRACK_ONLY = "crack_only"
    UTILITY_ONLY = "utility_only"
    CRACK_AND_UTILITY = "crack_and_utility"


class Label(IntEnum):
    """Per-cell class label written to labels.npy."""

    INTACT_CLAY = 0
    INTACT_SHELL = 1
    INTACT_FOUNDATION = 2
    CRACK_AIR = 3
    CRACK_WATER = 4
    PLASTIC_PIPE = 5
    CONCRETE_CONDUIT = 6
    METAL_PIPE = 7


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------


class MaterialProperties(BaseModel):
    """Physical properties of one material (reference/midpoint values)."""

    resistivity_ohm_m: float = Field(gt=0)
    velocity_m_s: float = Field(gt=0)
    density_kg_m3: float = Field(gt=0)


class MaterialLibrary(BaseModel):
    """Library of reference material properties for all classes."""

    clay_core: MaterialProperties
    shell_gravel: MaterialProperties
    foundation: MaterialProperties
    crack_air: MaterialProperties
    crack_water: MaterialProperties
    plastic_pipe: MaterialProperties
    concrete_conduit: MaterialProperties
    metal_pipe: MaterialProperties


# ---------------------------------------------------------------------------
# Dam-type geometry ratios (calibrated to West-California embankment practice)
# ---------------------------------------------------------------------------


class DamTypeGeometry(BaseModel):
    """Geometry ratios for one DamType.

    Attributes
    ----------
    core_crest_ratio : float
        Core top width as fraction of dam crest width. Only relevant for
        zoned / puddle-core. Homogeneous ignores it.
    core_slope : float
        Core side slope H:V. Small values → near-vertical core.
    has_core : bool
        Whether the dam type has a distinct core region.
    """

    core_crest_ratio: float = Field(gt=0, le=1)
    core_slope: float = Field(ge=0)
    has_core: bool


# ---------------------------------------------------------------------------
# Scenario / dam / anomaly configs
# ---------------------------------------------------------------------------


class DamConfig(BaseModel):
    """Geometric description of one dam cross-section."""

    dam_type: DamType
    height_m: float = Field(ge=10, le=25)
    crest_width_m: float = Field(ge=4, le=10)
    upstream_slope: float = Field(ge=2.5, le=3.5, description="H:V ratio")
    downstream_slope: float = Field(ge=2.0, le=3.0, description="H:V ratio")


class CrackConfig(BaseModel):
    """A longitudinal vertical crack piercing the 2-D section.

    Rendered as a parallelogram: width = ``aperture_mm`` / 1000, sheared by
    ``tilt_deg`` from vertical (positive tilts the top toward +x).
    """

    aperture_mm: float = Field(ge=1, le=300)
    depth_top_m: float = Field(ge=0)
    depth_bottom_m: float = Field(gt=0, le=20)
    tilt_deg: float = Field(default=0.0, ge=-60.0, le=60.0)
    fill: Literal["air", "water"]
    x_offset_m: float

    @model_validator(mode="after")
    def _depth_order(self) -> "CrackConfig":
        if self.depth_bottom_m <= self.depth_top_m:
            raise ValueError("depth_bottom_m must be greater than depth_top_m")
        return self


class UtilityConfig(BaseModel):
    """A single buried utility (pipe/conduit) in or near the dam."""

    utility_type: Literal["plastic_pipe", "concrete_conduit", "metal_pipe"]
    depth_m: float = Field(gt=0)
    diameter_m: float = Field(ge=0.1, le=2.0)
    x_position_m: float


class SaturationState(BaseModel):
    """One saturation level applied to a scenario."""

    state_id: int = Field(ge=0, le=2)
    phreatic_level: float = Field(ge=0, le=1, description="fraction of dam height")


class ScenarioConfig(BaseModel):
    """Full specification of one scenario (sampled per dataset item)."""

    scenario_id: str
    scenario_type: ScenarioType
    dam: DamConfig
    crack: CrackConfig | None = None
    utility: UtilityConfig | None = None
    saturation_states: list[SaturationState]
    seed: int

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ScenarioConfig":
        if self.scenario_type == ScenarioType.CRACK_ONLY:
            if self.crack is None or self.utility is not None:
                raise ValueError("CRACK_ONLY requires crack and no utility")
        elif self.scenario_type == ScenarioType.UTILITY_ONLY:
            if self.utility is None or self.crack is not None:
                raise ValueError("UTILITY_ONLY requires utility and no crack")
        elif self.scenario_type == ScenarioType.CRACK_AND_UTILITY:
            if self.crack is None or self.utility is None:
                raise ValueError("CRACK_AND_UTILITY requires both crack and utility")
        if len(self.saturation_states) < 1:
            raise ValueError("at least 1 saturation state required")
        return self


# ---------------------------------------------------------------------------
# Mesh + generation settings
# ---------------------------------------------------------------------------


class MeshConfig(BaseModel):
    """Mesh resolution and domain extent settings."""

    domain_side_margin_m: float = 20.0
    foundation_depth_m: float = 10.0
    max_area_dam_m2: float = 0.25
    max_area_anomaly_m2: float = 0.02
    max_area_far_m2: float = 2.0
    quality: float = 33.0


class GenerationConfig(BaseModel):
    """Dataset-generation-level knobs."""

    jitter_fraction: float = 0.15
    archie_saturation: float = 0.9
    phreatic_fractions: tuple[float, ...] = (1.0,)


# ---------------------------------------------------------------------------
# Default values (CA-calibrated West-California embankments)
# ---------------------------------------------------------------------------


DAM_TYPE_GEOMETRIES: dict[DamType, DamTypeGeometry] = {
    # Small agricultural/district embankments: single homogeneous clay/silt fill
    DamType.HOMOGENEOUS: DamTypeGeometry(
        core_crest_ratio=1.0, core_slope=0.0, has_core=False,
    ),
    # Modern zoned earthfill (Oroville/Castaic style): central clay core
    # with gravel shells. Core is roughly half the crest width, slopes mild.
    DamType.ZONED: DamTypeGeometry(
        core_crest_ratio=0.45, core_slope=0.4, has_core=True,
    ),
    # Historic puddle-core dams (San Andreas, old Calaveras): narrow
    # near-vertical clay core flanked by wider shells.
    DamType.PUDDLE_CORE: DamTypeGeometry(
        core_crest_ratio=0.20, core_slope=0.1, has_core=True,
    ),
}


_REFERENCE_MATERIALS = MaterialLibrary(
    clay_core=MaterialProperties(resistivity_ohm_m=50, velocity_m_s=1600, density_kg_m3=2000),
    shell_gravel=MaterialProperties(resistivity_ohm_m=500, velocity_m_s=450, density_kg_m3=1800),
    foundation=MaterialProperties(resistivity_ohm_m=25, velocity_m_s=1000, density_kg_m3=1900),
    crack_air=MaterialProperties(resistivity_ohm_m=20000, velocity_m_s=200, density_kg_m3=5),
    crack_water=MaterialProperties(resistivity_ohm_m=5, velocity_m_s=1450, density_kg_m3=1000),
    plastic_pipe=MaterialProperties(resistivity_ohm_m=30000, velocity_m_s=2200, density_kg_m3=1000),
    concrete_conduit=MaterialProperties(resistivity_ohm_m=1000, velocity_m_s=3500, density_kg_m3=2300),
    metal_pipe=MaterialProperties(resistivity_ohm_m=0.005, velocity_m_s=5500, density_kg_m3=7800),
)


class _Defaults(BaseModel):
    """Top-level container of all default configurations."""

    materials: MaterialLibrary = _REFERENCE_MATERIALS
    mesh: MeshConfig = MeshConfig()
    generation: GenerationConfig = GenerationConfig()


DEFAULTS = _Defaults()


# ---------------------------------------------------------------------------
# Property bounds (for validation)
# ---------------------------------------------------------------------------


RESISTIVITY_BOUNDS_OHM_M: tuple[float, float] = (1e-3, 1e5)
VELOCITY_BOUNDS_M_S: tuple[float, float] = (100.0, 6000.0)


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


UTILITY_LABEL: dict[str, Label] = {
    "plastic_pipe": Label.PLASTIC_PIPE,
    "concrete_conduit": Label.CONCRETE_CONDUIT,
    "metal_pipe": Label.METAL_PIPE,
}

UTILITY_MATERIAL_NAME: dict[str, str] = {
    "plastic_pipe": "plastic_pipe",
    "concrete_conduit": "concrete_conduit",
    "metal_pipe": "metal_pipe",
}

CRACK_LABEL: dict[str, Label] = {
    "air": Label.CRACK_AIR,
    "water": Label.CRACK_WATER,
}
