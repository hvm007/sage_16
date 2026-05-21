# Manifest

This file records the finalized sanitized artifact contents for upload.

## Top-level contents

- `data/` (contains only `data/v2/`)
- `rtl/`
- `scripts/`
- `docs/`
- `sim/`
- `.gitignore`
- `run_sim.bat`

## Reproducibility-critical subset

- `rtl/`: synthesizable SAGE-16 fabric RTL modules and wrappers
- `sim/`: simulation testbenches for functional validation, including quaternion vector rotation and CIFAR-10 smoke coverage
- `scripts/`: simulation, synthesis, MiniCNN, and CIFAR-10 export helper scripts
- `data/v2/`: bundled MiniCNN-v2 tensors and metadata for reproducible runs
- `run_sim.bat`: root-level simulation entrypoint

## Intentionally excluded from this package

- generated outputs such as `build/` and waveform dumps
- large synthesis output trees such as `reports/`
- large generated CIFAR `.mem` payloads and firmware C headers
- unrelated project areas (firmware, board files, notes, planning docs)

## Upload note

This package is sanitized so that legacy naming terms are removed from file names and file contents.
