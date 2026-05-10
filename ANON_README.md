# Anonymous RTL Artifact Guide

This package is prepared for double-blind RTL reproducibility review. It includes a sanitized SAGE-16 fabric implementation, simulation testbenches, and the minimal data payload required to rerun reported checks.

## 1) Prerequisites

- Windows command prompt (`cmd`) or PowerShell
- Icarus Verilog (`iverilog`, `vvp`) in `PATH`
- Python 3 (optional, only for regenerating `data/v2/*.mem`)
- Vivado (optional, only for synthesis reproduction)

## 2) Required quick validation

Run these commands from the artifact root:

```bat
run_sim.bat matmul
run_sim.bat minicnn-v2
```

Expected behavior for each command:

- finishes with `SUMMARY: pass=... fail=0`
- exits with `All requested testbenches finished.`

## 3) Additional simulation options

- all default non-MNIST benches:

```bat
run_sim.bat
```

- MiniCNN-v2 real-MNIST flow with bundled tensors:

```bat
run_sim.bat minicnn-v2-mnist
```

## 4) Optional data regeneration

To regenerate the bundled `data/v2/*.mem` payload:

```bat
python scripts\train_minicnn_v2.py
```

## 5) Optional synthesis reproduction

Single top:

```bat
vivado -mode batch -source scripts\synth_top.tcl -tclargs sage16_top 5.000 ooc
```

Batch sweep:

```bat
cmd /c scripts\synth_all.bat
```
