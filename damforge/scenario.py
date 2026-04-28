"""Scenario assembly — anomaly polygons and random scenario sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from damforge.config import (
    DAM_TYPE_GEOMETRIES,
    CrackConfig,
    DamConfig,
    DamType,
    GenerationConfig,
    SaturationState,
    ScenarioConfig,
    ScenarioType,
    UtilityConfig,
)
from damforge.dam import DamPolygons, build_dam_polygons


@dataclass(frozen=True)
class ScenarioGeometry:
    """Assembled geometry for one scenario (dam + optional anomalies)."""

    dam_polygons: DamPolygons
    crack_polygon: Polygon | None
    utility_polygon: Polygon | None


# ---------------------------------------------------------------------------
# Anomaly polygons
# ---------------------------------------------------------------------------


def build_crack_polygon(crack: CrackConfig, dam_cfg: DamConfig) -> Polygon:
    """Build a tilted parallelogram polygon for a longitudinal crack."""
    half = crack.aperture_mm / 1000.0 / 2.0
    x = crack.x_offset_m
    y_top, y_bot = -crack.depth_top_m, -crack.depth_bottom_m
    shear = (crack.depth_bottom_m - crack.depth_top_m) * np.tan(np.deg2rad(crack.tilt_deg))
    return Polygon(
        [
            (x - half + shear, y_top),
            (x + half + shear, y_top),
            (x + half, y_bot),
            (x - half, y_bot),
        ]
    )


def build_utility_polygon(util: UtilityConfig, n_sides: int = 32) -> Polygon:
    """Build a circular polygon approximating a buried utility cross-section.

    Parameters
    ----------
    util : UtilityConfig
    n_sides : int
        Polygon resolution for the circular cross-section.

    Returns
    -------
    Polygon
    """
    centre = Point(util.x_position_m, -util.depth_m)
    return centre.buffer(util.diameter_m / 2.0, quad_segs=max(n_sides // 4, 4))


def assemble_scenario_geometry(
    cfg: ScenarioConfig,
    mesh_cfg,
) -> ScenarioGeometry:
    """Build all geometric regions needed to mesh one scenario."""
    dam_polys = build_dam_polygons(cfg.dam, mesh_cfg)
    crack_poly = build_crack_polygon(cfg.crack, cfg.dam) if cfg.crack else None
    utility_poly = build_utility_polygon(cfg.utility) if cfg.utility else None
    return ScenarioGeometry(
        dam_polygons=dam_polys,
        crack_polygon=crack_poly,
        utility_polygon=utility_poly,
    )


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------


def _sample_dam(rng: np.random.Generator, dam_type: DamType) -> DamConfig:
    return DamConfig(
        dam_type=dam_type,
        height_m=float(rng.uniform(10.0, 25.0)),
        crest_width_m=float(rng.uniform(4.0, 10.0)),
        upstream_slope=float(rng.uniform(2.5, 3.5)),
        downstream_slope=float(rng.uniform(2.0, 3.0)),
    )


def _sample_crack(rng: np.random.Generator, dam_cfg: DamConfig) -> CrackConfig:
    """Sample a tilted, water-filled longitudinal crack inside the dam core."""
    # Vertical extent (top in shallow third, ≥4 m span, capped at ~50 ft).
    depth_top = float(rng.uniform(0.0, max(0.1, dam_cfg.height_m * 0.2)))
    max_bottom = min(dam_cfg.height_m - 0.5, 15.5)
    depth_bottom = float(rng.uniform(min(depth_top + 4.0, max_bottom), max_bottom))

    # Aperture biased to upper half of the realistic 1–300 mm range so ERT/
    # seismic resolve it. Half-width in metres for geometry calculations.
    aperture_mm = float(rng.uniform(50.0, 300.0))
    half = aperture_mm / 1000.0 / 2.0

    # Lateral budget: keep the crack inside the clay core (zoned/puddle) or
    # the central embankment band (homogeneous) so the PLC stays well-formed.
    geom = DAM_TYPE_GEOMETRIES[dam_cfg.dam_type]
    if geom.has_core:
        max_half = max(dam_cfg.crest_width_m * geom.core_crest_ratio / 2.0 - 0.1, 0.05)
    else:
        max_half = dam_cfg.crest_width_m * 0.25

    # Tilt up to 15°, but capped further so the sheared parallelogram fits
    # inside [-max_half, +max_half] when x_offset = 0.
    depth_span = depth_bottom - depth_top
    max_shear = max(max_half - half - 0.05, 0.0)
    tilt_cap = min(15.0, float(np.rad2deg(np.arctan(max_shear / depth_span))))
    tilt_deg = float(rng.uniform(0.0, tilt_cap)) * float(rng.choice([-1.0, 1.0]))

    # Position bottom edge so the entire parallelogram stays inside the core.
    shear = abs(depth_span * np.tan(np.deg2rad(tilt_deg)))
    slack = max_half - half - shear
    x_offset = float(rng.uniform(-slack, slack)) if slack > 0 else 0.0

    return CrackConfig(
        aperture_mm=aperture_mm,
        depth_top_m=depth_top,
        depth_bottom_m=depth_bottom,
        tilt_deg=tilt_deg,
        fill="water",
        x_offset_m=x_offset,
    )


def _sample_utility(rng: np.random.Generator, dam_cfg: DamConfig) -> UtilityConfig:
    utility_type = str(rng.choice(["plastic_pipe", "concrete_conduit", "metal_pipe"]))
    diameter = float(rng.uniform(0.2, 1.0))
    # Place utilities buried in the foundation layer (below the dam base) so
    # the polygon is always contained in a single meshable region. Assumes
    # MeshConfig.foundation_depth_m >= 4 m (default 10 m).
    depth = float(
        rng.uniform(dam_cfg.height_m + 1.0, dam_cfg.height_m + 4.0)
    )
    # Keep x within the embankment footprint so the utility stays near the dam.
    half_base = (
        dam_cfg.crest_width_m / 2.0
        + min(dam_cfg.upstream_slope, dam_cfg.downstream_slope) * dam_cfg.height_m
    )
    x_pos = float(rng.uniform(-half_base * 0.8, half_base * 0.8))
    return UtilityConfig(
        utility_type=utility_type,  # type: ignore[arg-type]
        depth_m=depth,
        diameter_m=diameter,
        x_position_m=x_pos,
    )


def _default_saturation_states(gen_cfg: GenerationConfig) -> list[SaturationState]:
    return [SaturationState(state_id=2, phreatic_level=1.0)]


def sample_scenario(
    scenario_id: str,
    scenario_type: ScenarioType,
    dam_type: DamType,
    seed: int,
    gen_cfg: GenerationConfig,
) -> ScenarioConfig:
    """Sample a random scenario of the given type.

    Parameters
    ----------
    scenario_id : str
        Unique scenario identifier (e.g. "scenario_0001").
    scenario_type : ScenarioType
    dam_type : DamType
    seed : int
        Per-scenario seed.
    gen_cfg : GenerationConfig

    Returns
    -------
    ScenarioConfig
    """
    rng = np.random.default_rng(seed)
    dam_cfg = _sample_dam(rng, dam_type)
    crack = (
        _sample_crack(rng, dam_cfg)
        if scenario_type in (ScenarioType.CRACK_ONLY, ScenarioType.CRACK_AND_UTILITY)
        else None
    )
    utility = (
        _sample_utility(rng, dam_cfg)
        if scenario_type in (ScenarioType.UTILITY_ONLY, ScenarioType.CRACK_AND_UTILITY)
        else None
    )
    return ScenarioConfig(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        dam=dam_cfg,
        crack=crack,
        utility=utility,
        saturation_states=_default_saturation_states(gen_cfg),
        seed=seed,
    )
