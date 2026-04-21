"""Dam geometry tests."""

from __future__ import annotations

import pytest

from damforge.config import DEFAULTS, DamConfig, DamType
from damforge.dam import build_dam_polygons, phreatic_y


def _cfg(dam_type: DamType) -> DamConfig:
    return DamConfig(
        dam_type=dam_type, height_m=15.0, crest_width_m=6.0,
        upstream_slope=3.0, downstream_slope=2.5,
    )


@pytest.mark.parametrize("dam_type", list(DamType))
def test_polygon_areas_positive(dam_type: DamType):
    polys = build_dam_polygons(_cfg(dam_type), DEFAULTS.mesh)
    assert polys.embankment.area > 0
    assert polys.foundation.area > 0
    if polys.clay_core is not None:
        assert polys.clay_core.area > 0
    if polys.shell is not None:
        assert polys.shell.area > 0


def test_homogeneous_has_no_core():
    polys = build_dam_polygons(_cfg(DamType.HOMOGENEOUS), DEFAULTS.mesh)
    assert polys.clay_core is None
    assert polys.shell is None


def test_zoned_core_inside_embankment():
    polys = build_dam_polygons(_cfg(DamType.ZONED), DEFAULTS.mesh)
    assert polys.clay_core is not None
    assert polys.embankment.contains(polys.clay_core.buffer(-1e-6))


def test_puddle_core_narrower_than_zoned():
    puddle = build_dam_polygons(_cfg(DamType.PUDDLE_CORE), DEFAULTS.mesh)
    zoned = build_dam_polygons(_cfg(DamType.ZONED), DEFAULTS.mesh)
    assert puddle.clay_core.area < zoned.clay_core.area


def test_phreatic_y_monotonic():
    cfg = _cfg(DamType.ZONED)
    assert phreatic_y(cfg, 0.3) < phreatic_y(cfg, 0.6) < phreatic_y(cfg, 0.85)
    assert phreatic_y(cfg, 1.0) == 0.0
    assert phreatic_y(cfg, 0.0) == -cfg.height_m


def test_slope_ratio_honored():
    cfg = _cfg(DamType.HOMOGENEOUS)
    polys = build_dam_polygons(cfg, DEFAULTS.mesh)
    xs = [x for x, _ in polys.embankment.exterior.coords]
    expected_us_toe = -(cfg.crest_width_m / 2 + cfg.upstream_slope * cfg.height_m)
    expected_ds_toe = cfg.crest_width_m / 2 + cfg.downstream_slope * cfg.height_m
    assert min(xs) == pytest.approx(expected_us_toe)
    assert max(xs) == pytest.approx(expected_ds_toe)
