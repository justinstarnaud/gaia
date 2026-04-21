"""Dam cross-section polygon geometry.

Pure geometry — no PyGIMLi, no matplotlib. Returns shapely polygons that
`mesh.py` converts into PyGIMLi PLCs.

Coordinate convention: x=0 at dam centre-line, y=0 at crest (negative down).
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from damforge.config import (
    DAM_TYPE_GEOMETRIES,
    DamConfig,
    DamType,
    MeshConfig,
)


@dataclass(frozen=True)
class DamPolygons:
    """Geometric regions of a dam cross-section.

    Attributes
    ----------
    embankment : Polygon
        Full dam body (clay core for homogeneous, or core+shells union).
    clay_core : Polygon | None
        Central clay-core region (None for homogeneous).
    shell : Polygon | None
        Shell region (embankment minus clay core). None for homogeneous.
    foundation : Polygon
        Foundation layer beneath the embankment.
    domain_bounds : tuple[float, float, float, float]
        (x_min, y_min, x_max, y_max) of the full modelling domain (m).
    """

    embankment: Polygon
    clay_core: Polygon | None
    shell: Polygon | None
    foundation: Polygon
    domain_bounds: tuple[float, float, float, float]


def build_dam_polygons(dam_cfg: DamConfig, mesh_cfg: MeshConfig) -> DamPolygons:
    """Build the polygonal regions of a dam cross-section.

    Parameters
    ----------
    dam_cfg : DamConfig
    mesh_cfg : MeshConfig

    Returns
    -------
    DamPolygons
    """
    h = dam_cfg.height_m
    wc = dam_cfg.crest_width_m
    us = dam_cfg.upstream_slope
    ds = dam_cfg.downstream_slope

    x_us_toe = -(wc / 2.0 + us * h)
    x_ds_toe = wc / 2.0 + ds * h
    y_base = -h
    y_bottom = -h - mesh_cfg.foundation_depth_m
    x_min = x_us_toe - mesh_cfg.domain_side_margin_m
    x_max = x_ds_toe + mesh_cfg.domain_side_margin_m

    embankment = Polygon(
        [
            (x_us_toe, y_base),
            (-wc / 2.0, 0.0),
            (wc / 2.0, 0.0),
            (x_ds_toe, y_base),
        ]
    )

    foundation = Polygon(
        [
            (x_min, y_bottom),
            (x_max, y_bottom),
            (x_max, y_base),
            (x_min, y_base),
        ]
    )

    geom = DAM_TYPE_GEOMETRIES[dam_cfg.dam_type]
    clay_core: Polygon | None = None
    shell: Polygon | None = None

    if geom.has_core:
        core_crest = wc * geom.core_crest_ratio
        # Core widens with depth per core_slope (H:V).
        core_base_half = core_crest / 2.0 + geom.core_slope * h
        core_top_half = core_crest / 2.0
        clay_core = Polygon(
            [
                (-core_top_half, 0.0),
                (core_top_half, 0.0),
                (core_base_half, y_base),
                (-core_base_half, y_base),
            ]
        ).intersection(embankment)
        shell = embankment.difference(clay_core)

    return DamPolygons(
        embankment=embankment,
        clay_core=clay_core if clay_core and not clay_core.is_empty else None,
        shell=shell if shell and not shell.is_empty else None,
        foundation=foundation,
        domain_bounds=(x_min, y_bottom, x_max, y_base),
    )


def phreatic_y(dam_cfg: DamConfig, phreatic_fraction: float) -> float:
    """Return the y-coordinate of the horizontal phreatic surface.

    Parameters
    ----------
    dam_cfg : DamConfig
    phreatic_fraction : float
        0 → foundation, 1 → crest.

    Returns
    -------
    float
        y-coordinate in metres (negative, since crest is at y=0).
    """
    return -dam_cfg.height_m * (1.0 - phreatic_fraction)


def dam_type_summary(dam_type: DamType) -> str:
    """Human-readable description of a dam type."""
    g = DAM_TYPE_GEOMETRIES[dam_type]
    return (
        f"{dam_type.value}: has_core={g.has_core}, "
        f"core_crest_ratio={g.core_crest_ratio}, core_slope={g.core_slope}"
    )
