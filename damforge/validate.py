"""Pre-save validation for generated scenarios.

Runs the 5 rules defined in CLAUDE.md. Returns a ValidationReport; callers
decide whether to proceed based on `.ok`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from damforge.config import (
    RESISTIVITY_BOUNDS_OHM_M,
    VELOCITY_BOUNDS_M_S,
    CRACK_LABEL,
    UTILITY_LABEL,
    Label,
    ScenarioConfig,
)
from damforge.properties import PropertyArrays


@dataclass
class ValidationReport:
    """Outcome of scenario validation."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_scenario(
    cfg: ScenarioConfig,
    labels: np.ndarray,
    per_state_props: list[PropertyArrays],
) -> ValidationReport:
    """Validate a generated scenario against the 5 rules.

    Parameters
    ----------
    cfg : ScenarioConfig
    labels : np.ndarray, shape (n_cells,)
    per_state_props : list[PropertyArrays]
        One entry per saturation state (ordered by state_id).

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    # Rule 1: injected anomaly classes must appear in labels.
    expected: set[int] = set()
    if cfg.crack is not None:
        expected.add(CRACK_LABEL[cfg.crack.fill].value)
    if cfg.utility is not None:
        expected.add(UTILITY_LABEL[cfg.utility.utility_type].value)
    present = set(int(x) for x in np.unique(labels))
    missing = expected - present
    if missing:
        names = [Label(m).name for m in missing]
        report.add_error(f"Missing anomaly labels: {names}")

    # Rule 2 & 3: property bounds across all states.
    rho_lo, rho_hi = RESISTIVITY_BOUNDS_OHM_M
    v_lo, v_hi = VELOCITY_BOUNDS_M_S
    for i, props in enumerate(per_state_props):
        if (props.resistivity_ohm_m < rho_lo).any() or (props.resistivity_ohm_m > rho_hi).any():
            report.add_error(
                f"state_{i}: resistivity out of bounds [{rho_lo}, {rho_hi}]"
            )
        if (props.velocity_m_s < v_lo).any() or (props.velocity_m_s > v_hi).any():
            report.add_error(
                f"state_{i}: velocity out of bounds [{v_lo}, {v_hi}]"
            )

    # Rule 4: state_0 and state_2 resistivity must differ.
    if len(per_state_props) >= 3:
        if np.allclose(
            per_state_props[0].resistivity_ohm_m,
            per_state_props[2].resistivity_ohm_m,
        ):
            report.add_error("state_0 and state_2 resistivity arrays are identical")

    # Rule 5: crack/utility resistivity overlap warning (>10% of anomaly cells).
    if cfg.crack is not None and cfg.utility is not None:
        crack_lbl = CRACK_LABEL[cfg.crack.fill].value
        util_lbl = UTILITY_LABEL[cfg.utility.utility_type].value
        state1 = per_state_props[min(1, len(per_state_props) - 1)]
        crack_rho = state1.resistivity_ohm_m[labels == crack_lbl]
        util_rho = state1.resistivity_ohm_m[labels == util_lbl]
        if crack_rho.size and util_rho.size:
            lo = max(crack_rho.min(), util_rho.min())
            hi = min(crack_rho.max(), util_rho.max())
            if hi > lo:
                n_overlap = int(((crack_rho >= lo) & (crack_rho <= hi)).sum())
                total = crack_rho.size + util_rho.size
                if n_overlap / total > 0.10:
                    report.add_warning(
                        f"Crack/utility resistivity overlap in "
                        f"{n_overlap}/{total} anomaly cells"
                    )

    return report
