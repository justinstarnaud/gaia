"""PyGIMLi mesh construction from scenario geometry.

Region markers are set to match `Label` enum IDs so downstream property
assignment can use cell markers directly.
"""

from __future__ import annotations

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from shapely.geometry import Polygon

from damforge.config import (
    UTILITY_LABEL,
    CRACK_LABEL,
    DamType,
    Label,
    MeshConfig,
    ScenarioConfig,
)
from damforge.scenario import ScenarioGeometry


def _poly_verts(poly: Polygon) -> list[tuple[float, float]]:
    """Return exterior vertices of a shapely Polygon (no closing duplicate)."""
    coords = list(poly.exterior.coords)
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


def build_mesh(
    cfg: ScenarioConfig,
    geometry: ScenarioGeometry,
    mesh_cfg: MeshConfig,
) -> pg.Mesh:
    """Triangulate the full scenario domain into a labelled PyGIMLi mesh.

    Parameters
    ----------
    cfg : ScenarioConfig
    geometry : ScenarioGeometry
    mesh_cfg : MeshConfig

    Returns
    -------
    pg.Mesh
        Cells carry `marker()` equal to the corresponding `Label.value`.
    """
    dp = geometry.dam_polygons

    # Domain background = foundation (marker covers outer frame below ground
    # and acts as the "outer" region).
    x_min, y_min, x_max, y_base = dp.domain_bounds
    outer = mt.createPolygon(
        [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_base),
            (x_min, y_base),
        ],
        isClosed=True,
        marker=Label.INTACT_FOUNDATION.value,
        area=mesh_cfg.max_area_far_m2,
    )
    plcs = [outer]

    # Embankment: homogeneous gets single CLAY marker; zoned/puddle get
    # shell + clay-core polygons.
    if cfg.dam.dam_type == DamType.HOMOGENEOUS or dp.clay_core is None:
        plcs.append(
            mt.createPolygon(
                _poly_verts(dp.embankment),
                isClosed=True,
                marker=Label.INTACT_CLAY.value,
                area=mesh_cfg.max_area_dam_m2,
            )
        )
    else:
        if dp.shell is not None:
            # Shell may be a MultiPolygon (US + DS). Add each piece.
            shell_geoms = (
                dp.shell.geoms if dp.shell.geom_type == "MultiPolygon" else [dp.shell]
            )
            for g in shell_geoms:
                plcs.append(
                    mt.createPolygon(
                        _poly_verts(g),
                        isClosed=True,
                        marker=Label.INTACT_SHELL.value,
                        area=mesh_cfg.max_area_dam_m2,
                    )
                )
        plcs.append(
            mt.createPolygon(
                _poly_verts(dp.clay_core),
                isClosed=True,
                marker=Label.INTACT_CLAY.value,
                area=mesh_cfg.max_area_dam_m2,
            )
        )

    # Anomalies
    if geometry.crack_polygon is not None and cfg.crack is not None:
        plcs.append(
            mt.createPolygon(
                _poly_verts(geometry.crack_polygon),
                isClosed=True,
                marker=CRACK_LABEL[cfg.crack.fill].value,
                area=mesh_cfg.max_area_anomaly_m2,
            )
        )

    if geometry.utility_polygon is not None and cfg.utility is not None:
        plcs.append(
            mt.createPolygon(
                _poly_verts(geometry.utility_polygon),
                isClosed=True,
                marker=UTILITY_LABEL[cfg.utility.utility_type].value,
                area=mesh_cfg.max_area_anomaly_m2,
            )
        )

    plc = mt.mergePLC(plcs)
    mesh = mt.createMesh(plc, quality=mesh_cfg.quality, smooth=[1, 10])

    if mesh.cellCount() == 0:
        raise RuntimeError("Mesh produced zero cells")
    return mesh


def cell_centers(mesh: pg.Mesh) -> np.ndarray:
    """Return (n_cells, 2) array of cell-centre (x, y) coordinates in m."""
    return np.array([[c.center().x(), c.center().y()] for c in mesh.cells()])


def cell_markers(mesh: pg.Mesh) -> np.ndarray:
    """Return (n_cells,) int array of cell region markers."""
    return np.array([c.marker() for c in mesh.cells()], dtype=np.int32)
