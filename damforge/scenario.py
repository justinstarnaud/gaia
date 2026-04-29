"""Scenario assembly — anomaly polygons and random scenario sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon

from damforge.config import (
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
    """Crack polygon anchored on the upstream face of the embankment.

    The top-left vertex sits on the upstream face at ``depth_top_m``; the
    crack extends inward by ``aperture_mm`` and downward to ``depth_bottom_m``,
    with the bottom shifted further into the dam by ``tilt_deg``.
    """
    aperture = crack.aperture_mm / 1000.0
    half_crest = dam_cfg.crest_width_m / 2.0
    us = dam_cfg.upstream_slope
    x_face_top = -half_crest - us * crack.depth_top_m
    y_top, y_bot = -crack.depth_top_m, -crack.depth_bottom_m
    inward = (crack.depth_bottom_m - crack.depth_top_m) * np.tan(
        np.deg2rad(crack.tilt_deg)
    )
    return Polygon(
        [
            (x_face_top, y_top),
            (x_face_top + aperture, y_top),
            (x_face_top + aperture + inward, y_bot),
            (x_face_top + inward, y_bot),
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
    """Sample a water-filled longitudinal crack rooted on the upstream face."""
    hard_max_bottom = min(dam_cfg.height_m - 0.5, 20.0)
    depth_top_hi = max(0.0, hard_max_bottom - 4.0)
    # Mix shallow-start (upper third) and mid-depth (seepage initiation behind
    # the phreatic line) so the dataset spans realistic crack positions.
    if rng.random() < 0.5:
        depth_top = float(rng.uniform(0.0, min(dam_cfg.height_m * 0.2, depth_top_hi)))
    else:
        lo = min(dam_cfg.height_m * 0.4, depth_top_hi)
        hi = min(dam_cfg.height_m * 0.7, depth_top_hi)
        depth_top = float(rng.uniform(lo, hi)) if hi > lo else lo
    max_bottom = min(hard_max_bottom, depth_top + 6.0)
    depth_bottom = float(rng.uniform(depth_top + 4.0, max_bottom))

    aperture_mm = float(rng.uniform(50.0, 300.0))
    tilt_deg = float(rng.uniform(15.0, 60.0))

    return CrackConfig(
        aperture_mm=aperture_mm,
        depth_top_m=depth_top,
        depth_bottom_m=depth_bottom,
        tilt_deg=tilt_deg,
        fill="water",
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
