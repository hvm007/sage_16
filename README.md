# SAGE-16

**A 16-DSP Reconfigurable Fabric for Edge Vision and Attitude Kernels**

SAGE-16 is a compact Artix-7 edge coprocessor that time-multiplexes three kernel
families — **3×3 convolution**, **matrix multiplication**, and **quaternion
Hamilton products** — on one shared 4×4 MAC array (16 DSP48E1 slices), with
**single-cycle kernel switching** via a 2-bit mode register and **no bitstream
reload**.

This repository contains the complete RTL, testbenches, training/export
scripts, and data needed to reproduce the results of our APCCAS 2026 paper:

> Harshvardhan Mishra and Madhav Rao,
> *"SAGE-16: A 16-DSP Reconfigurable Fabric for Edge Vision and Attitude Kernels,"*
> IEEE Asia Pacific Conference on Circuits and Systems (APCCAS), 2026.

## Highlights

| Metric | Value |
|---|---|
| DSP usage vs three separate accelerators | 16 vs 48 (**-66.7%**) |
| Post-route Fmax (xc7a35tcpg236-1, OOC) | **159.6 MHz** |
| Throughput | 1.436 GMAC/s (conv), 0.928 GMAC/s (matmul/quat) |
| Kernel switch latency | **1 cycle** (no bitstream reload) |
| MNIST (MiniCNN-v2, full 10,000 images, RTL) | **95.04% top-1**, 49.09M bit-exact checks, 0 mismatches |
| CIFAR-10 (VGG-BN v5, full 10,000-image export) | **87.69% fixed-point** (91.84% float) |
| Measured Cortex-M4 speedups (STM32G491) | up to 38.6× scalar / 29.7× CMSIS-DSP `SMLAD` |

## Repository layout

```
rtl/        Synthesizable RTL: PE, 4x4 MAC fabric, kernel controllers,
            shared top (sage16_top), MMIO + PicoRV32-PCPI wrappers
sim/        Self-checking testbenches for every claim in the paper
scripts/    Simulation runner, Vivado synthesis flow, training/export scripts
data/v2/    MiniCNN-v2 MNIST payload: INT4 weights + full 10,000-image
            test set + golden predictions (ready to run, no download needed)
docs/       CIFAR-10 train/export flow notes
```

## Quick start (60 seconds)

Requirements: [Icarus Verilog](https://steveicarus.github.io/iverilog/) 11+
(`iverilog`/`vvp` on `PATH`). Windows: run from `cmd` or PowerShell.

```bat
run_sim.bat matmul   & REM 304 output cells, overflow stress
run_sim.bat conv     & REM Sobel/Laplacian/random kernels, 464 cells
run_sim.bat quat     & REM Hamilton products, 304 components
```

Every testbench is self-checking and ends with `SUMMARY: pass=N fail=0`.

On Linux/macOS the pattern for any testbench is:

```bash
iverilog -g2012 -o build/sim.vvp \
  rtl/pe.v rtl/sage16_4x4_mac.v rtl/matmul_sage16.v rtl/sage16_top.v \
  sim/tb_reconfig_pairs.v
vvp build/sim.vvp
```

## Reproducing the paper's claims

| Paper claim | Command | Expected result |
|---|---|---|
| Matmul kernel correctness | `run_sim.bat matmul` | 304 cells, `fail=0` |
| Conv3×3 kernel correctness | `run_sim.bat conv` | 464 cells, `fail=0` |
| Quaternion kernel correctness | `run_sim.bat quat` | 304 components, `fail=0` |
| Quaternion vector rotation (22 cycles) | `run_sim.bat quat-vrot` | `fail=0` |
| 8×8 / 16×16 tiled matmul | `run_sim.bat tiled` | `fail=0` |
| 1-cycle back-to-back switching | `run_sim.bat b2b` | `fail=0`, 1-cycle gap |
| All 6 directed mode transitions | `run_sim.bat pairs` | `transitions: 6/6`, `pass=192 fail=0` |
| MMIO / PCPI host integration | `run_sim.bat mmio_top` / `pcpi_top` | `fail=0` |
| Madgwick attitude filter (100 IMU steps) | `run_sim.bat madgwick` | `fail=0` |
| **MNIST 95.04% top-1, full 10k, bit-exact** | `run_sim.bat minicnn-v2-mnist` | `top-1: 9504/10000`, 0 mismatches |
| CIFAR-10 sampled RTL pass | see [CIFAR-10](#cifar-10-full-flow) below | `pred matches export`, `fail=0` |

Notes on the MNIST run: the full 10,000-image test set and trained INT4
weights are bundled in `data/v2/`, and `tb_minicnn_v2.v` defaults to
`N_IMG=10000`. The run checks every intermediate tensor of every image
bit-exactly (~49M checks) and takes several hours in Icarus. For a quick look,
pass a smaller image count:

```bash
iverilog -g2012 -DREAL_MNIST -Ptb_minicnn_v2.N_IMG=200 -o build/mnist200.vvp \
  rtl/pe.v rtl/sage16_4x4_mac.v rtl/matmul_sage16.v rtl/sage16_top.v \
  sim/tb_minicnn_v2.v
vvp build/mnist200.vvp
```

## CIFAR-10 full flow

The CIFAR-10 payload (~150 MB for 10k images) is not committed; it is
regenerated deterministically by two commands (requires Python 3 with
`torch`/`torchvision`; training takes a few GPU-hours, export takes minutes):

```bash
# 1) Train the float VGG-BN model (seed fixed for reproducibility)
python scripts/train_minicnn_cifar10.py --arch v5 --float-only --epochs 60 \
  --lr 0.05 --batch 128 --w_max 15 --conv2_shift 3 --weight_decay 0.0005 \
  --label_smoothing 0.05 --export 0 --seed 1 \
  --save-checkpoint checkpoints/cifar_v5_float60_seed1.pt

# 2) Fold BN, calibrate fixed-point shifts, export the full 10k test set
python scripts/train_minicnn_cifar10.py --arch v5 \
  --load-checkpoint checkpoints/cifar_v5_float60_seed1.pt \
  --epochs 0 --relaxed-epochs 0 --quant-epochs 0 --lr 0.05 --batch 128 \
  --w_max 15 --conv2_shift 3 --weight_decay 0.0005 --label_smoothing 0.05 \
  --shift-sweep --sweep-conv2 4:10 --sweep-pool1 0:4 --sweep-pool2 0:4 \
  --sweep-gap 0:5 --sweep-samples 5000 --export 10000
```

Step 2 writes `data/cifar10_v5/` and prints the summary; `info.txt` should
report `export_acc: 0.8769` (87.69% on the full 10,000-image test set,
`best_float_acc: 0.9184`). Then run the sampled RTL validation:

```bash
iverilog -g2012 -DREAL_CIFAR10_V5 -Ptb_cifar10_v5.N_IMG=16 \
  -o build/cifar16.vvp \
  rtl/pe.v rtl/sage16_4x4_mac.v rtl/matmul_sage16.v rtl/sage16_top.v \
  sim/tb_cifar10_v5.v
vvp build/cifar16.vvp
```

The testbench replays the exported fixed-point inference on the SAGE-16
fabric and checks each prediction against the exported `preds.txt`
(`pred matches export : N / N`). Since the RTL is bit-exact against the
export, the full-set 87.69% transfers to the fabric by construction.
Each CIFAR image is ~48M fabric cycles, so budget simulation time accordingly
(`N_IMG=2` via `run_sim.bat cifar10-v5-smoke` for a quick check).

## Synthesis (Vivado)

Results in the paper use Vivado 2025.1, out-of-context, xc7a35tcpg236-1:

```bat
REM single top (module, clock period ns, mode)
vivado -mode batch -source scripts\synth_top.tcl -tclargs sage16_top 5.000 ooc

REM all paper-relevant tops (shared fabric + separate baselines + wrappers)
cmd /c scripts\synth_all.bat
```

`scripts/synth_all.bat` also synthesizes the three kernel-specific
accelerators separately — that is the internal baseline behind the paper's
66.7% DSP / 6.8% LUT resource-sharing comparison.

## Cortex-M4 baselines

Embedded baselines were measured on an STM32G491 (Cortex-M4F, 170 MHz,
`arm-none-eabi-gcc -O3`, DWT cycle counter), using the same integer tensors
as the RTL testbenches, in scalar and CMSIS-DSP `SMLAD` variants. The
firmware tree is not part of this artifact; the matched operand headers can
be regenerated with `scripts/export_v2_mem_to_c.py` and
`scripts/export_cifar10_mem_to_c.py`.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{mishra2026sage16,
  author    = {Mishra, Harshvardhan and Rao, Madhav},
  title     = {{SAGE-16}: A 16-{DSP} Reconfigurable Fabric for Edge Vision and Attitude Kernels},
  booktitle = {IEEE Asia Pacific Conference on Circuits and Systems (APCCAS)},
  year      = {2026}
}
```
