# GAIA

Subsurface crack detection in
embankment dams — NASA Tournament Lab / Bureau of Reclamation dam-safety
competition.

## Blocks

| # | Block | Status | Docs |
|---|-------|--------|------|
| 1 | Dam modelling (synthetic dataset generation) | ✅ | [damforge/README.md](damforge/README.md) |
| 2 | ERT forward modelling (PyGIMLi) | — | |
| 3 | Seismic forward modelling (PyGIMLi / SpecFEM) | — | |
| 4–5 | Joint inversion (SimPEG) | — | |
| 6 | ML fusion | — | |

## Setup

PyGIMLi has no macOS ARM wheel on PyPI; install it via conda first, then
bootstrap the uv environment:

```bash
conda create -n damforge python=3.12
conda install -n damforge -c gimli -c conda-forge pygimli
make setup
```

## Tests

```bash
make test
```
