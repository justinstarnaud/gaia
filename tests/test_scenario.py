"""Scenario sampling + saturation logic tests (no PyGIMLi dependency)."""

from __future__ import annotations

import numpy as np
import pytest

from damforge.config import (
    DEFAULTS,
    CRACK_LABEL,
    UTILITY_LABEL,
    DamType,
    Label,
    ScenarioType,
)
from damforge.properties import apply_saturation, assign_base_properties
from damforge.scenario import sample_scenario


def test_sample_scenario_consistency():
    for st in ScenarioType:
        for dt in DamType:
            cfg = sample_scenario(
                scenario_id="s", scenario_type=st, dam_type=dt,
                seed=0, gen_cfg=DEFAULTS.generation,
            )
            if st == ScenarioType.CRACK_ONLY:
                assert cfg.crack is not None and cfg.utility is None
            elif st == ScenarioType.UTILITY_ONLY:
                assert cfg.utility is not None and cfg.crack is None
            else:
                assert cfg.crack is not None and cfg.utility is not None
            assert len(cfg.saturation_states) == 3


def _synthetic_labels() -> np.ndarray:
    # 3 foundation cells, 2 shell, 2 crack_air, 2 plastic_pipe
    return np.array(
        [
            Label.INTACT_FOUNDATION, Label.INTACT_FOUNDATION, Label.INTACT_FOUNDATION,
            Label.INTACT_SHELL, Label.INTACT_SHELL,
            Label.CRACK_AIR, Label.CRACK_AIR,
            Label.PLASTIC_PIPE, Label.PLASTIC_PIPE,
        ],
        dtype=np.int32,
    )


def test_saturation_affects_soil_resistivity():
    labels = _synthetic_labels()
    cell_y = np.full(labels.shape, -10.0)  # all below any phreatic at y=-5
    rng = np.random.default_rng(0)
    base = assign_base_properties(labels, DEFAULTS.materials, rng, 0.0)
    sat = apply_saturation(
        base, labels, cell_y, phreatic_y_m=-5.0,
        archie_saturation=0.9, library=DEFAULTS.materials,
    )
    soil = np.isin(labels, [Label.INTACT_FOUNDATION.value, Label.INTACT_SHELL.value])
    # Archie's law: ρ_sat = ρ_dry / S² with S=0.9 → factor ~1.23
    assert np.all(sat.resistivity_ohm_m[soil] > base.resistivity_ohm_m[soil])


def test_utilities_never_change_across_saturation():
    labels = _synthetic_labels()
    cell_y = np.full(labels.shape, -10.0)
    rng = np.random.default_rng(0)
    base = assign_base_properties(labels, DEFAULTS.materials, rng, 0.0)
    util_mask = labels == Label.PLASTIC_PIPE.value
    for y in [-5.0, -1.0, -20.0]:
        sat = apply_saturation(
            base, labels, cell_y, phreatic_y_m=y,
            archie_saturation=0.9, library=DEFAULTS.materials,
        )
        assert np.array_equal(sat.resistivity_ohm_m[util_mask], base.resistivity_ohm_m[util_mask])
        assert np.array_equal(sat.velocity_m_s[util_mask], base.velocity_m_s[util_mask])


def test_air_crack_below_phreatic_becomes_water():
    labels = _synthetic_labels()
    cell_y = np.full(labels.shape, -10.0)
    rng = np.random.default_rng(0)
    base = assign_base_properties(labels, DEFAULTS.materials, rng, 0.0)
    sat = apply_saturation(
        base, labels, cell_y, phreatic_y_m=-5.0,
        archie_saturation=0.9, library=DEFAULTS.materials,
    )
    crack_mask = labels == Label.CRACK_AIR.value
    water = DEFAULTS.materials.crack_water
    assert np.allclose(sat.resistivity_ohm_m[crack_mask], water.resistivity_ohm_m)
    assert np.allclose(sat.velocity_m_s[crack_mask], water.velocity_m_s)


def test_air_crack_above_phreatic_unchanged():
    labels = _synthetic_labels()
    cell_y = np.full(labels.shape, -1.0)  # all above phreatic at y=-5
    rng = np.random.default_rng(0)
    base = assign_base_properties(labels, DEFAULTS.materials, rng, 0.0)
    sat = apply_saturation(
        base, labels, cell_y, phreatic_y_m=-5.0,
        archie_saturation=0.9, library=DEFAULTS.materials,
    )
    crack_mask = labels == Label.CRACK_AIR.value
    assert np.allclose(sat.resistivity_ohm_m[crack_mask], base.resistivity_ohm_m[crack_mask])
