# DamForge

Synthetic labelled geophysical datasets of embankment dams, for training a joint
ERT + seismic crack-detection model. Block 1 of the gaia pipeline.

## What it produces

For each scenario, DamForge builds a 2-D cross-section of an embankment dam,
injects zero or one crack and zero or one utility, triangulates the domain, and
renders the material-property fields at 3 saturation states.

- **3 dam types** (CA-calibrated to West-California practice): homogeneous,
  zoned (Oroville/Castaic style), puddle-core (San Andreas / old Calaveras style).
- **3 scenario types**: `crack_only`, `utility_only`, `crack_and_utility`.
- **3 saturation states** per scenario (phreatic level at 30 %, 60 %, 85 % of dam height).
- Per-cell arrays for resistivity (Ω·m), P-wave velocity (m/s), density (kg/m³),
  and integer class labels.

## Install

PyGIMLi has no macOS ARM wheel on PyPI; install it first into a conda env, then
let `uv` point at that env's Python.

```bash
conda create -n damforge python=3.12
conda install -n damforge -c gimli -c conda-forge pygimli
make setup
```

## Run

```bash
uv run damforge --n-scenarios 100 --seed 42
```

Or from Python:

```python
from pathlib import Path
from damforge.generate import generate_dataset

generate_dataset(n_scenarios=100, output_dir=Path("./output"), seed=42)
```

## Output layout

```
output/
  scenario_0001/
    config.json                # ScenarioConfig (Pydantic, round-trippable)
    labels.npy                 # (n_cells,) int32
    mesh.bms                   # PyGIMLi mesh
    state_0/                   # phreatic = 0.30 × dam height
      resistivity.npy          # (n_cells,) float32, Ω·m
      velocity.npy             # (n_cells,) float32, m/s
      density.npy              # (n_cells,) float32, kg/m³
    state_1/                   # phreatic = 0.60
    state_2/                   # phreatic = 0.85
    plots/
      triptych.png             # resistivity | velocity | labels (state 1)
      saturation_trajectory.png
  property_space.png           # dataset-level (log ρ, V) scatter
```

## Labels

| ID | Name |
|----|------|
| 0  | `INTACT_CLAY` |
| 1  | `INTACT_SHELL` |
| 2  | `INTACT_FOUNDATION` |
| 3  | `CRACK_AIR` |
| 4  | `CRACK_WATER` |
| 5  | `PLASTIC_PIPE` |
| 6  | `CONCRETE_CONDUIT` |
| 7  | `METAL_PIPE` |

(Crack orientation — `transverse` vs `longitudinal` — is stored as a field on
`CrackConfig`, not as a separate label class.)

## Coordinate system

- `x` : horizontal, `x = 0` at dam centre-line.
- `y` : vertical,   `y = 0` at crest, negative downward.

## Saturation physics

Below the horizontal phreatic surface:

- **Soil cells** (clay core, shell, foundation): Archie's law,
  `ρ_sat = ρ_dry / S²` with S ≈ 0.9. Velocity bumped +10 % to approximate the
  saturated Vp shift.
- **Crack cells** initially marked `CRACK_AIR`: overridden to the `CRACK_WATER`
  material below the phreatic surface.
- **Utility cells**: invariant across saturation states — this is the core
  discriminator the downstream ML model is trained to exploit.

## Tests

```bash
make test
```

Covers config validation, dam polygon geometry, saturation-state logic, and
utility invariance across states.
