"""All plotting for DamForge. Only module that imports matplotlib.

- `triptych(scenario_dir)` — resistivity | velocity | labels for state_1.
- `saturation_trajectory(scenario_dir)` — (log ρ, V) scatter across all 3 states.
- `property_space(scenario_dirs, output_path)` — dataset-level cluster plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg

from damforge.config import CRACK_LABEL, UTILITY_LABEL, Label
from damforge.export import load_config, load_labels, load_mesh, load_state
from damforge.mesh import cell_centers
from damforge.scenario import (
    build_crack_polygon,
    build_utility_polygon,
)


def _draw_mesh_field(ax, mesh: pg.Mesh, values: np.ndarray, **kw):
    """Render a per-cell field on a PyGIMLi mesh via pg.show."""
    pg.show(mesh, data=values, ax=ax, showMesh=False, colorBar=True, **kw)


def _outline_anomalies(ax, cfg) -> None:
    """Draw white-dashed outlines of the crack and utility polygons."""
    if cfg.crack is not None:
        poly = build_crack_polygon(cfg.crack, cfg.dam)
        x, y = poly.exterior.xy
        ax.plot(x, y, "--", color="white", linewidth=1.2)
    if cfg.utility is not None:
        poly = build_utility_polygon(cfg.utility)
        x, y = poly.exterior.xy
        ax.plot(x, y, "--", color="white", linewidth=1.2)


def triptych(scenario_dir: Path) -> Path:
    """Render the 3-panel plot (resistivity | velocity | labels) for the last state."""
    cfg = load_config(scenario_dir)
    mesh = load_mesh(scenario_dir)
    labels = load_labels(scenario_dir)
    state_id = cfg.saturation_states[-1].state_id
    props = load_state(scenario_dir, state_id=state_id)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _draw_mesh_field(
        axes[0], mesh, props.resistivity_ohm_m,
        cMap="viridis", logScale=True, label="Resistivity [Ω·m]",
    )
    axes[0].set_title(f"Resistivity (state {state_id})")
    _outline_anomalies(axes[0], cfg)

    _draw_mesh_field(
        axes[1], mesh, props.velocity_m_s,
        cMap="magma", label="Velocity [m/s]",
    )
    axes[1].set_title(f"Velocity (state {state_id})")
    _outline_anomalies(axes[1], cfg)

    _draw_mesh_field(
        axes[2], mesh, labels.astype(float),
        cMap="tab10", label="Label",
    )
    axes[2].set_title("Label map")
    _outline_anomalies(axes[2], cfg)

    fig.suptitle(f"{cfg.scenario_id} — {cfg.scenario_type.value} / {cfg.dam.dam_type.value}")
    fig.tight_layout()
    out = scenario_dir / "plots" / "triptych.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def saturation_trajectory(scenario_dir: Path) -> Path:
    """Plot (log ρ, V) for all cells at each saturation state.

    Crack cells should visibly move across states; utility cells should stay
    fixed — the core scientific claim of the dataset.
    """
    cfg = load_config(scenario_dir)
    labels = load_labels(scenario_dir)
    states = [load_state(scenario_dir, s.state_id) for s in cfg.saturation_states]

    fig, ax = plt.subplots(figsize=(8, 6))
    markers = ["o", "s", "^"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    for i, props in enumerate(states):
        ax.scatter(
            props.resistivity_ohm_m,
            props.velocity_m_s,
            s=3, alpha=0.35, marker=markers[i], color=colors[i],
            label=f"state {i} (φ={cfg.saturation_states[i].phreatic_level:.2f})",
        )

    # Highlight anomaly cells
    anomaly_mask = np.zeros_like(labels, dtype=bool)
    if cfg.crack is not None:
        anomaly_mask |= labels == CRACK_LABEL[cfg.crack.fill].value
    if cfg.utility is not None:
        anomaly_mask |= labels == UTILITY_LABEL[cfg.utility.utility_type].value
    for i, props in enumerate(states):
        ax.scatter(
            props.resistivity_ohm_m[anomaly_mask],
            props.velocity_m_s[anomaly_mask],
            s=18, marker=markers[i], edgecolor="black",
            facecolor=colors[i], linewidth=0.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Resistivity [Ω·m]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title(f"{cfg.scenario_id} — saturation trajectory")
    ax.legend()
    fig.tight_layout()
    out = scenario_dir / "plots" / "saturation_trajectory.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def property_space(
    scenario_dirs: list[Path], output_path: Path, state_id: int | None = None
) -> Path:
    if state_id is None:
        state_id = load_config(scenario_dirs[0]).saturation_states[-1].state_id
    """Dataset-level scatter of (log ρ, V) across all scenarios, by label."""
    rho_by_label: dict[int, list[float]] = {}
    v_by_label: dict[int, list[float]] = {}

    for sd in scenario_dirs:
        labels = load_labels(sd)
        props = load_state(sd, state_id=state_id)
        for lbl in np.unique(labels):
            mask = labels == lbl
            rho_by_label.setdefault(int(lbl), []).extend(
                props.resistivity_ohm_m[mask].tolist()
            )
            v_by_label.setdefault(int(lbl), []).extend(
                props.velocity_m_s[mask].tolist()
            )

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for lbl, rho in rho_by_label.items():
        rho_a = np.asarray(rho)
        v_a = np.asarray(v_by_label[lbl])
        # Subsample to keep the plot legible
        if rho_a.size > 2000:
            idx = np.random.default_rng(0).choice(rho_a.size, 2000, replace=False)
            rho_a, v_a = rho_a[idx], v_a[idx]
        ax.scatter(
            rho_a, v_a, s=4, alpha=0.4, color=cmap(lbl % 10),
            label=Label(lbl).name,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Resistivity [Ω·m]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title(f"Property space (state {state_id}, n_scenarios={len(scenario_dirs)})")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
