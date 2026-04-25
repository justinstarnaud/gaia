"""Dataset generation entry point + CLI.

Runs the full pipeline for `n_scenarios` items:
    sample config → build geometry → mesh → properties (3 states) →
    validate → save → per-scenario plots.
Failures for individual scenarios are logged and skipped; the dataset-level
property-space plot is rendered at the end.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from damforge.config import (
    DEFAULTS,
    DamType,
    ScenarioConfig,
    ScenarioType,
)
from damforge.dam import phreatic_y
from damforge.export import save_scenario
from damforge.mesh import build_mesh, cell_centers, cell_markers
from damforge.properties import apply_saturation, assign_base_properties
from damforge.scenario import assemble_scenario_geometry, sample_scenario
from damforge.validate import validate_scenario
from damforge.visualize import property_space, saturation_trajectory, triptych

logger = logging.getLogger("damforge")


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------


_SCENARIO_TYPES: tuple[ScenarioType, ...] = tuple(ScenarioType)
_DAM_TYPES: tuple[DamType, ...] = tuple(DamType)


def _plan_scenarios(n: int) -> list[tuple[ScenarioType, DamType]]:
    """Produce n (scenario_type, dam_type) pairs distributed evenly.

    Types cycle through their cross-product to keep counts balanced.
    """
    combos = [(s, d) for s in _SCENARIO_TYPES for d in _DAM_TYPES]
    return [combos[i % len(combos)] for i in range(n)]


# ---------------------------------------------------------------------------
# Single-scenario pipeline
# ---------------------------------------------------------------------------


def _run_scenario(cfg: ScenarioConfig):
    """Run the full pipeline for one scenario, returning in-memory artifacts.

    Returns
    -------
    tuple of (cfg, mesh, labels, per_state).
    """
    rng = np.random.default_rng(cfg.seed)

    geometry = assemble_scenario_geometry(cfg, DEFAULTS.mesh)
    mesh = build_mesh(cfg, geometry, DEFAULTS.mesh)
    labels = cell_markers(mesh)
    centres = cell_centers(mesh)
    cell_y = centres[:, 1]

    base_props = assign_base_properties(
        labels, DEFAULTS.materials, rng, DEFAULTS.generation.jitter_fraction
    )

    per_state = [
        apply_saturation(
            base_props,
            labels,
            cell_y,
            phreatic_y(cfg.dam, state.phreatic_level),
            DEFAULTS.generation.archie_saturation,
            DEFAULTS.materials,
        )
        for state in cfg.saturation_states
    ]

    report = validate_scenario(cfg, labels, per_state)
    for w in report.warnings:
        logger.warning("[%s] %s", cfg.scenario_id, w)
    if not report.ok:
        raise RuntimeError(
            f"Validation failed for {cfg.scenario_id}: {report.errors}"
        )

    return cfg, mesh, labels, per_state


def _persist_scenario(output_dir: Path, cfg, mesh, labels, per_state) -> Path:
    scenario_dir = save_scenario(output_dir, cfg, mesh, labels, per_state)
    triptych(scenario_dir)
    saturation_trajectory(scenario_dir)
    return scenario_dir


# ---------------------------------------------------------------------------
# Dataset driver
# ---------------------------------------------------------------------------


def generate_dataset(
    n_scenarios: int,
    output_dir: Path | None = None,
    seed: int = 42,
    write: bool = True,
):
    """Generate a full labelled dataset.

    Parameters
    ----------
    n_scenarios : int
    output_dir : Path | None
        Required when `write=True`; ignored when `write=False`.
    seed : int
        Base seed. Per-scenario seeds are `seed + i`.
    write : bool
        If True (default), persist each scenario to `output_dir` and return
        the list of written scenario directories. If False, return a list of
        in-memory `(cfg, mesh, labels, per_state)` tuples.

    Returns
    -------
    list[Path] when write=True, else list[tuple[ScenarioConfig, pg.Mesh, np.ndarray, list]]
    """
    if write:
        if output_dir is None:
            raise ValueError("output_dir is required when write=True")
        output_dir.mkdir(parents=True, exist_ok=True)

    plan = _plan_scenarios(n_scenarios)

    written: list[Path] = []
    results: list = []
    for i, (scenario_type, dam_type) in enumerate(tqdm(plan, desc="scenarios")):
        scenario_id = f"scenario_{i + 1:04d}"
        try:
            cfg = sample_scenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                dam_type=dam_type,
                seed=seed + i,
                gen_cfg=DEFAULTS.generation,
            )
            artifacts = _run_scenario(cfg)
            if write:
                written.append(_persist_scenario(output_dir, *artifacts))
            else:
                results.append(artifacts)
        except Exception as exc:  # noqa: BLE001 — we must skip & log, not crash
            logger.exception("Scenario %s failed: %s", scenario_id, exc)

    if write:
        if written:
            sample = written[: min(50, len(written))]
            property_space(sample, output_dir / "property_space.png")
        logger.info(
            "Generated %d/%d scenarios in %s", len(written), n_scenarios, output_dir
        )
        return written

    logger.info("Generated %d/%d scenarios (in-memory)", len(results), n_scenarios)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Console-script entry point wired via [project.scripts]."""
    parser = argparse.ArgumentParser(prog="damforge")
    parser.add_argument("--n-scenarios", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    generate_dataset(args.n_scenarios, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    _cli()
