# RTL Artifact Guide

This package is prepared for double-blind RTL reproducibility review. It includes a sanitized SAGE-16 fabric implementation, simulation testbenches, and the minimal data payload required to rerun reported checks.

## 1) Prerequisites

- Windows command prompt (`cmd`) or PowerShell
- Icarus Verilog (`iverilog`, `vvp`) in `PATH`
- Python 3 (optional, only for regenerating `data/v2/*.mem` or CIFAR-10 exports)
- Vivado (optional, only for synthesis reproduction)

## 2) Required quick validation

Run these commands from the artifact root:

```bat
run_sim.bat matmul
run_sim.bat minicnn-v2
run_sim.bat quat-vrot
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

- CIFAR-10 v5 smoke path after generating or copying `data/cifar10_v5/*.mem` locally:

```bat
run_sim.bat cifar10-v5-smoke
```

## 4) Optional data regeneration

To regenerate the bundled `data/v2/*.mem` payload:

```bat
python scripts\train_minicnn_v2.py
```

To train/export CIFAR-10 artifacts locally:

```bat
python scripts\train_minicnn_cifar10.py --arch v5 --smoke --synthetic
```

See `docs/CIFAR10_TRAIN_EXPORT.md` for the full CIFAR export flow. Large CIFAR `.mem` payloads and generated C headers are intentionally excluded from Git.

## 5) Optional synthesis reproduction

Single top:

```bat
vivado -mode batch -source scripts\synth_top.tcl -tclargs sage16_top 5.000 ooc
```

Batch sweep:

```bat
cmd /c scripts\synth_all.bat
```
