# CIFAR-10 MiniCNN Train + Export

This flow adds a CIFAR-10 variant of the existing MiniCNN-v2 style pipeline, with
hex16 `.mem` artifacts compatible with the int16/int32 software/hardware flow.

## Scripts

- `scripts/train_minicnn_cifar10.py`
- `scripts/export_cifar10_mem_to_c.py`

## Artifact layout

Training/export writes to `data/cifar10_v1/`:

- weights: `conv1_k.mem`, `conv2_k.mem`, `w1.mem`, `w2.mem`
- samples: `images.mem`, `labels.mem`
- metadata: `info.txt`, `labels.txt`, `preds.txt`

Conventions:

- all `.mem` entries are 16-bit hex words (`to_hex16`, two's complement for negatives),
- images are exported in CHW order (`C` then `H` then `W`) per sample,
- `w2.mem` has 12 rows (`10` real classes + `2` zero-padded rows), matching existing style.

## Local usage (Windows/Linux)

From repo root:

```bash
# quick smoke (no download, synthetic data)
python scripts/train_minicnn_cifar10.py --smoke --synthetic

# real CIFAR-10 short run
python scripts/train_minicnn_cifar10.py --epochs 5 --subset-train 5000 --subset-test 1000 --export 64

# full run (tune epochs as needed)
python scripts/train_minicnn_cifar10.py --epochs 80 --export 200

# optional header export for firmware bring-up
python scripts/export_cifar10_mem_to_c.py --n-img 16
```

## Optional GPU Usage

The same commands can be run on any Python environment with PyTorch/TorchVision installed. Generated artifacts should be copied into `data/cifar10_<arch>/` when needed. Large generated headers are intentionally not tracked.
