"""Disk I/O for generated scenarios. ALL filesystem access for the package
goes through this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pygimli as pg

from damforge.config import ScenarioConfig
from damforge.properties import PropertyArrays


def save_scenario(
    output_dir: Path,
    cfg: ScenarioConfig,
    mesh: pg.Mesh,
    labels: np.ndarray,
    per_state_props: list[PropertyArrays],
) -> Path:
    """Write one scenario to disk in the layout defined in CLAUDE.md.

    Parameters
    ----------
    output_dir : Path
        Parent directory (e.g. `./data`). A subdirectory named after
        `cfg.scenario_id` is created inside it.
    cfg : ScenarioConfig
    mesh : pg.Mesh
    labels : np.ndarray, shape (n_cells,)
    per_state_props : list[PropertyArrays]
        Ordered by state_id.

    Returns
    -------
    Path
        The scenario directory that was written.
    """
    scenario_dir = output_dir / cfg.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    (scenario_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    np.save(scenario_dir / "labels.npy", labels.astype(np.int32))
    mesh.save(str(scenario_dir / "mesh.bms"))

    for state, props in zip(cfg.saturation_states, per_state_props):
        sdir = scenario_dir / f"state_{state.state_id}"
        sdir.mkdir(exist_ok=True)
        np.save(sdir / "resistivity.npy", props.resistivity_ohm_m.astype(np.float32))
        np.save(sdir / "velocity.npy", props.velocity_m_s.astype(np.float32))
        np.save(sdir / "density.npy", props.density_kg_m3.astype(np.float32))

    (scenario_dir / "plots").mkdir(exist_ok=True)
    return scenario_dir


def load_config(scenario_dir: Path) -> ScenarioConfig:
    """Read and validate a scenario's config.json."""
    raw = json.loads((scenario_dir / "config.json").read_text())
    return ScenarioConfig.model_validate(raw)


def load_labels(scenario_dir: Path) -> np.ndarray:
    return np.load(scenario_dir / "labels.npy")


def load_state(scenario_dir: Path, state_id: int) -> PropertyArrays:
    sdir = scenario_dir / f"state_{state_id}"
    return PropertyArrays(
        resistivity_ohm_m=np.load(sdir / "resistivity.npy"),
        velocity_m_s=np.load(sdir / "velocity.npy"),
        density_kg_m3=np.load(sdir / "density.npy"),
    )


def load_mesh(scenario_dir: Path) -> pg.Mesh:
    return pg.load(str(scenario_dir / "mesh.bms"))
