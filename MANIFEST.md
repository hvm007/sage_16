# Manifest

This file records the finalized sanitized artifact contents for upload.

## Top-level contents

- `data/` (contains only `data/v2/`)
- `rtl/`
- `scripts/`
- `sim/`
- `.gitignore`
- `ANON_README.md`
- `ANON_UPLOAD_MANIFEST.md`
- `run_sim.bat`

## Reproducibility-critical subset

- `rtl/`: synthesizable SAGE-16 fabric RTL modules and wrappers
- `sim/`: simulation testbenches for functional validation
- `scripts/`: simulation and synthesis helper scripts
- `data/v2/`: bundled MiniCNN-v2 tensors and metadata for reproducible runs
- `run_sim.bat`: root-level simulation entrypoint

## Intentionally excluded from this package

- generated outputs such as `build/` and waveform dumps
- large synthesis output trees such as `reports/`
- unrelated project areas (firmware, board files, notes, planning docs)

## Upload note

This package is sanitized so that legacy naming terms are removed from file names and file contents.
