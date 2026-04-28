"""Config-model validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from damforge.config import (
    DEFAULTS,
    CrackConfig,
    DamConfig,
    DamType,
    RESISTIVITY_BOUNDS_OHM_M,
    SaturationState,
    ScenarioConfig,
    ScenarioType,
    UtilityConfig,
    VELOCITY_BOUNDS_M_S,
)


def _dam() -> DamConfig:
    return DamConfig(
        dam_type=DamType.ZONED,
        height_m=15.0,
        crest_width_m=6.0,
        upstream_slope=3.0,
        downstream_slope=2.5,
    )


def _states() -> list[SaturationState]:
    return [
        SaturationState(state_id=0, phreatic_level=0.30),
        SaturationState(state_id=1, phreatic_level=0.60),
        SaturationState(state_id=2, phreatic_level=0.85),
    ]


def _crack() -> CrackConfig:
    return CrackConfig(
        aperture_mm=100.0, depth_top_m=1.0, depth_bottom_m=8.0,
        tilt_deg=10.0, fill="water", x_offset_m=0.5,
    )


def _utility() -> UtilityConfig:
    return UtilityConfig(
        utility_type="plastic_pipe", depth_m=12.0, diameter_m=0.4, x_position_m=3.0,
    )


def test_crack_only_rejects_utility():
    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="x", scenario_type=ScenarioType.CRACK_ONLY,
            dam=_dam(), crack=_crack(), utility=_utility(),
            saturation_states=_states(), seed=1,
        )


def test_utility_only_requires_utility():
    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="x", scenario_type=ScenarioType.UTILITY_ONLY,
            dam=_dam(), saturation_states=_states(), seed=1,
        )


def test_crack_and_utility_requires_both():
    cfg = ScenarioConfig(
        scenario_id="x", scenario_type=ScenarioType.CRACK_AND_UTILITY,
        dam=_dam(), crack=_crack(), utility=_utility(),
        saturation_states=_states(), seed=1,
    )
    assert cfg.crack is not None and cfg.utility is not None


def test_requires_three_saturation_states():
    with pytest.raises(ValidationError):
        ScenarioConfig(
            scenario_id="x", scenario_type=ScenarioType.CRACK_ONLY,
            dam=_dam(), crack=_crack(),
            saturation_states=_states()[:2], seed=1,
        )


def test_crack_depth_order():
    with pytest.raises(ValidationError):
        CrackConfig(
            aperture_mm=10.0, depth_top_m=5.0, depth_bottom_m=3.0,
            fill="water", x_offset_m=0.0,
        )


def test_crack_aperture_bounds():
    for ap in (1.0, 300.0):
        CrackConfig(
            aperture_mm=ap, depth_top_m=0.0, depth_bottom_m=2.0,
            fill="water", x_offset_m=0.0,
        )
    for ap in (0.5, 301.0):
        with pytest.raises(ValidationError):
            CrackConfig(
                aperture_mm=ap, depth_top_m=0.0, depth_bottom_m=2.0,
                fill="water", x_offset_m=0.0,
            )


def test_reference_materials_within_bounds():
    lib = DEFAULTS.materials
    rho_lo, rho_hi = RESISTIVITY_BOUNDS_OHM_M
    v_lo, v_hi = VELOCITY_BOUNDS_M_S
    for name in lib.model_fields:
        m = getattr(lib, name)
        assert rho_lo <= m.resistivity_ohm_m <= rho_hi, name
        assert v_lo <= m.velocity_m_s <= v_hi, name
