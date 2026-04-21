"""Per-cell property assignment and saturation-state logic.

- `assign_base_properties` produces the dry-state (resistivity, velocity,
  density) arrays by looking up each cell's marker in the material library
  and applying ±jitter_fraction Gaussian noise per scenario.
- `apply_saturation` overlays a phreatic-level-dependent transformation:
    * soil cells below phreatic: Archie's law ρ_sat = ρ_dry / S²
    * crack cells below phreatic: overridden to water-filled material
    * utility cells: never change (the core discriminator)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damforge.config import (
    RESISTIVITY_BOUNDS_OHM_M,
    VELOCITY_BOUNDS_M_S,
    Label,
    MaterialLibrary,
    MaterialProperties,
)


# Cells that behave as "soil" under saturation (Archie's law applies).
_SOIL_LABELS: set[int] = {
    Label.INTACT_CLAY.value,
    Label.INTACT_SHELL.value,
    Label.INTACT_FOUNDATION.value,
}

# Cells that represent utilities (never change under saturation).
_UTILITY_LABELS: set[int] = {
    Label.PLASTIC_PIPE.value,
    Label.CONCRETE_CONDUIT.value,
    Label.METAL_PIPE.value,
}


@dataclass(frozen=True)
class PropertyArrays:
    """Per-cell property arrays for one saturation state.

    All arrays have shape (n_cells,).
    """

    resistivity_ohm_m: np.ndarray
    velocity_m_s: np.ndarray
    density_kg_m3: np.ndarray


def _label_to_material(
    label: int, library: MaterialLibrary
) -> MaterialProperties:
    """Map a Label value to its reference MaterialProperties."""
    mapping = {
        Label.INTACT_CLAY.value: library.clay_core,
        Label.INTACT_SHELL.value: library.shell_gravel,
        Label.INTACT_FOUNDATION.value: library.foundation,
        Label.CRACK_AIR.value: library.crack_air,
        Label.CRACK_WATER.value: library.crack_water,
        Label.PLASTIC_PIPE.value: library.plastic_pipe,
        Label.CONCRETE_CONDUIT.value: library.concrete_conduit,
        Label.METAL_PIPE.value: library.metal_pipe,
    }
    return mapping[label]


def assign_base_properties(
    labels: np.ndarray,
    library: MaterialLibrary,
    rng: np.random.Generator,
    jitter_fraction: float,
) -> PropertyArrays:
    """Assign dry-state per-cell properties.

    Each cell's value is drawn from a Gaussian centred on the reference
    material property with std = jitter_fraction × reference. Values are
    clipped to remain positive.

    Parameters
    ----------
    labels : np.ndarray, shape (n_cells,)
        Integer Label value per cell.
    library : MaterialLibrary
    rng : np.random.Generator
    jitter_fraction : float
        Gaussian std as a fraction of the reference value (e.g. 0.15).

    Returns
    -------
    PropertyArrays
    """
    n = labels.shape[0]
    resistivity = np.empty(n, dtype=np.float64)
    velocity = np.empty(n, dtype=np.float64)
    density = np.empty(n, dtype=np.float64)

    for lbl in np.unique(labels):
        mat = _label_to_material(int(lbl), library)
        mask = labels == lbl
        k = int(mask.sum())
        rho_lo, rho_hi = RESISTIVITY_BOUNDS_OHM_M
        v_lo, v_hi = VELOCITY_BOUNDS_M_S
        # Leave 10% headroom so saturation effects don't push properties
        # out of the validity bounds downstream.
        resistivity[mask] = np.clip(
            rng.normal(mat.resistivity_ohm_m, jitter_fraction * mat.resistivity_ohm_m, k),
            rho_lo * 1.1, rho_hi * 0.9,
        )
        velocity[mask] = np.clip(
            rng.normal(mat.velocity_m_s, jitter_fraction * mat.velocity_m_s, k),
            v_lo * 1.1, v_hi * 0.9,
        )
        density[mask] = np.clip(
            rng.normal(mat.density_kg_m3, jitter_fraction * mat.density_kg_m3, k),
            mat.density_kg_m3 * 0.1, None,
        )

    return PropertyArrays(
        resistivity_ohm_m=resistivity,
        velocity_m_s=velocity,
        density_kg_m3=density,
    )


def apply_saturation(
    base: PropertyArrays,
    labels: np.ndarray,
    cell_y: np.ndarray,
    phreatic_y_m: float,
    archie_saturation: float,
    library: MaterialLibrary,
) -> PropertyArrays:
    """Apply phreatic-level saturation effects to base properties.

    Parameters
    ----------
    base : PropertyArrays
        Dry-state properties (not mutated).
    labels : np.ndarray, shape (n_cells,)
    cell_y : np.ndarray, shape (n_cells,)
        y-coordinate of each cell centre (m, negative below crest).
    phreatic_y_m : float
        y-coordinate of the horizontal phreatic surface (m).
    archie_saturation : float
        S in Archie's law ρ_sat = ρ_dry / S². Typical 0.9.
    library : MaterialLibrary

    Returns
    -------
    PropertyArrays
        New arrays for this saturation state.
    """
    resistivity = base.resistivity_ohm_m.copy()
    velocity = base.velocity_m_s.copy()
    density = base.density_kg_m3.copy()

    below = cell_y < phreatic_y_m

    # Soil: Archie's law on resistivity, small velocity bump from Vp↑ in
    # saturated media (approximate: +10%).
    soil_mask = below & np.isin(labels, list(_SOIL_LABELS))
    if soil_mask.any():
        resistivity[soil_mask] = resistivity[soil_mask] / (archie_saturation ** 2)
        velocity[soil_mask] = velocity[soil_mask] * 1.10

    # Crack cells below phreatic: override to water-filled material.
    crack_mask = below & (labels == Label.CRACK_AIR.value)
    if crack_mask.any():
        w = library.crack_water
        resistivity[crack_mask] = w.resistivity_ohm_m
        velocity[crack_mask] = w.velocity_m_s
        density[crack_mask] = w.density_kg_m3

    # Utilities are explicitly untouched (no-op, just documenting invariant).
    _ = np.isin(labels, list(_UTILITY_LABELS))

    return PropertyArrays(
        resistivity_ohm_m=resistivity,
        velocity_m_s=velocity,
        density_kg_m3=density,
    )
