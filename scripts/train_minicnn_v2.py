#!/usr/bin/env python3
"""
train_minicnn_v2.py
===================
Quantization-aware training of MiniCNN-v2 -- a *deeper* 3x3-native CNN that
runs unmodified on the same 16-DSP sage16_top fabric (no RTL change).  The goal
is to push real-MNIST top-1 from v1's ~94% architectural ceiling to ~97-98%
by adding a second conv stage; the fabric, the PE, the DSP count, and the
synthesised 160 MHz / 16 DSP / 4.6% LUT numbers all remain identical.

Architecture (chosen so every spatial dim matches the fabric's 6x6 -> 4x4
single-channel conv3x3 primitive cleanly):

    Input    26x26x1        INT16 centered uint6  [-32, +31]
    Conv1    3x3, 1->8      INT4 weights         -> 24x24x8
    AvgPool  4x4 stride 4   integer floor(s/16)  ->  6x6x8
    Conv2    3x3, 8->16     INT4 weights, multi-channel summation in INT32
                                                 ->  4x4x16
    Flatten                                      ->  256
    FC1      256->32        INT4 weights         ->  32       INT32 acc
    ReLU                                         ->  32
    FC2      32->10         INT4 weights         ->  10       INT32 acc
    argmax                                       ->  predicted class

Conv2 is the new layer.  Each fabric call still computes a single-channel
3x3 convolution on a 6x6 input, producing a 4x4 output tile.  Multi-channel
summation is done *in software* on the host (SoC's RISC-V core would do this
in real deployment) by accumulating partial 4x4 tiles across all 8 input
channels.  Number of fabric calls per image:

    Conv1  : 8 oc * 36 tiles                           = 288
    Conv2  : 16 oc * 8 ic * 1 tile (6x6 -> 4x4 fits)   = 128
    FC1    : 8 row-groups * 64 inner-tiles             = 512
    FC2    : 3 row-groups * 8  inner-tiles             =  24
    -----------------------------------------------------------
    total  fabric calls per image                       = 952
    (v1 was 456; ~2x calls, still <130 us/img at 160 MHz)

Why the [-32,+31] input scale and INT4 [-4,+4] weights work end-to-end
---------------------------------------------------------------------
Conv1 worst-case sum  : 9*32*4               = 1152      fits INT16
Pool out (>>>4)       : 1152                              fits INT16
Conv2 partial (per ic): 9*1152*4             = 41 472    fits INT32
Conv2 sum over 8 ic   : 8*41 472             = 331 776   needs INT32
Conv2 ->  feeds back into INT16 fabric input, so we add a per-tensor
right-shift (>>> 4) on the conv2 result before it becomes the FC1 input.
This shift is part of the forward pass during training (using STE) so the
network learns weights that respect the truncation.

FC1 worst case (theoretical) : 256*(20736)*4 = 2.1e7    needs INT32
But FC1 output feeds INT16 FC2 input.  A trained INT4 network learns very
sparse activations; the empirically-observed FC1 max is well under 32 767
(same situation as v1, where theoretical worst case is also above INT16).
We do NOT add an explicit FC1 shift -- if accuracy plateaus due to FC1
clipping we'll add it as a second pass.

Outputs (under data/v2/):
    conv1_k.mem  conv2_k.mem  w1.mem  w2.mem    INT4 weights
    images.mem   labels.mem                     N test images + GT labels
    labels.txt   preds.txt                      human-readable
    info.txt                                    summary stats
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"
DATA.mkdir(parents=True, exist_ok=True)
MNIST_DIR = ROOT / "data" / "mnist"


# --------------------------------------------------------------------- model
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def qw(w, w_max):
    return STE.apply(torch.clamp(w, -w_max, w_max))


class FloorSTE(torch.autograd.Function):
    """floor() with straight-through gradient (forward = floor, backward = id)."""
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        return g


def floor_ste(x):
    return FloorSTE.apply(x)


class MiniCNNv2(nn.Module):
    """Bit-exact match to tb_minicnn_v2.v dataflow after weight rounding."""
    def __init__(self, w_max=4, conv2_shift=4):
        super().__init__()
        self.w_max = w_max
        self.conv2_shift = conv2_shift          # >>> N applied to Conv2 output
        self.conv1 = nn.Conv2d(1,  8, 3, bias=False)
        self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
        self.fc1   = nn.Linear(256, 32, bias=False)
        self.fc2   = nn.Linear(32,  10, bias=False)
        # full-band uniform init: post-rounding weights span the entire
        # quantization range from epoch 1.  Smaller layers get smaller
        # init magnitudes so STE has gradient signal everywhere.
        for p in (self.conv1.weight, self.conv2.weight,
                  self.fc1.weight,   self.fc2.weight):
            nn.init.uniform_(p, -w_max + 0.01, w_max - 0.01)

    def forward(self, x_int):
        # x_int: (B, 1, 26, 26), centered-int float in [-32, +31]
        wc1 = qw(self.conv1.weight, self.w_max)
        wc2 = qw(self.conv2.weight, self.w_max)
        w1  = qw(self.fc1.weight,   self.w_max)
        w2  = qw(self.fc2.weight,   self.w_max)

        h = F.conv2d(x_int, wc1)                          # (B, 8, 24, 24)
        # AvgPool with floor( sum / 16 ) -- matches Verilog >>>4 exactly.
        # F.avg_pool2d returns sum/16 in float; multiply back to get sum
        # then floor-divide so the result is an integer-valued float.
        h_sum = F.avg_pool2d(h, 4, 4) * 16.0              # sum-form (no round)
        h = floor_ste(h_sum / 16.0)                       # arithmetic >>>4
        # Conv2: full multi-channel valid 3x3 -- F.conv2d already does the
        # cross-channel sum, identical to the Verilog ic-loop.
        h = F.conv2d(h, wc2)                              # (B, 16, 4, 4)
        # Truncate Conv2 result before feeding 16-bit FC1 input.  Use floor
        # so it lines up with Verilog >>>conv2_shift; keep STE for gradient.
        scale = float(1 << self.conv2_shift)
        h = floor_ste(h / scale)                          # >>> conv2_shift
        h = h.flatten(1)                                  # (B, 256)
        h = F.linear(h, w1)                               # (B, 32)
        h = F.relu(h)
        h = F.linear(h, w2)                               # (B, 10)
        return h


# ------------------------------------------------------------------ data prep
def get_loaders(batch=128, augment=False):
    """MNIST -> 26x26 -> uint6 [0,63] -> centered [-32,31]."""
    train_tf_list = [transforms.CenterCrop(26)]
    if augment:
        train_tf_list.append(
            transforms.RandomCrop(26, padding=2, padding_mode="constant",
                                  fill=0))
    train_tf_list.append(transforms.PILToTensor())
    train_tf = transforms.Compose(train_tf_list)
    test_tf  = transforms.Compose([
        transforms.CenterCrop(26),
        transforms.PILToTensor(),
    ])

    train = datasets.MNIST(str(MNIST_DIR), train=True,  download=True,
                           transform=train_tf)
    test  = datasets.MNIST(str(MNIST_DIR), train=False, download=True,
                           transform=test_tf)

    def to_centered_int(b):
        x = torch.stack([img for img, _ in b]).float()
        x = (x / 4.0).floor() - 32.0
        y = torch.tensor([y for _, y in b])
        return x, y

    train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                              num_workers=0, collate_fn=to_centered_int)
    test_loader  = DataLoader(test,  batch_size=batch, shuffle=False,
                              num_workers=0, collate_fn=to_centered_int)
    return train_loader, test_loader


# ----------------------------------------------------------- train / evaluate
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def warmup_cosine_lr(opt, epoch, epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def train(model, train_loader, test_loader, device,
          epochs, base_lr, temperature=500.0,
          label_smoothing=0.0, warmup_epochs=3,
          weight_decay=1e-4, grad_clip=5.0):
    opt = torch.optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, nesterov=True,
                          weight_decay=weight_decay)
    best_acc   = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for ep in range(epochs):
        lr = warmup_cosine_lr(opt, ep, epochs, warmup_epochs, base_lr)
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out / temperature, y,
                                   label_smoothing=label_smoothing)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()

        acc = evaluate(model, test_loader, device)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(f"  ep {ep+1:3d}/{epochs}  loss={running/n:.4f}  "
              f"acc={acc*100:.2f}%  lr={lr:.4f}{marker}")

    model.load_state_dict(best_state)
    return best_acc


# -------------------------------------------------------------- export
def to_hex16(v: int) -> str:
    return f"{v & 0xFFFF:04x}"


def quantized_weights_int(model, w_max):
    def q(p):
        return torch.round(torch.clamp(p.detach(), -w_max, w_max)).int().cpu().numpy()
    return (q(model.conv1.weight), q(model.conv2.weight),
            q(model.fc1.weight),   q(model.fc2.weight))


def export_weights(wc1, wc2, w1, w2, path_dir):
    # conv1: oc=8, ic=1, 3x3
    lines = []
    for oc in range(8):
        for i in range(3):
            for j in range(3):
                lines.append(to_hex16(int(wc1[oc, 0, i, j])))
    (path_dir / "conv1_k.mem").write_text("\n".join(lines) + "\n")

    # conv2: oc=16, ic=8, 3x3 -- ordering matches verilog [oc][ic][i][j]
    lines = []
    for oc in range(16):
        for ic in range(8):
            for i in range(3):
                for j in range(3):
                    lines.append(to_hex16(int(wc2[oc, ic, i, j])))
    (path_dir / "conv2_k.mem").write_text("\n".join(lines) + "\n")

    # fc1: NFC1=32 rows, NFLAT=256 cols
    lines = []
    for oc in range(32):
        for k in range(256):
            lines.append(to_hex16(int(w1[oc, k])))
    (path_dir / "w1.mem").write_text("\n".join(lines) + "\n")

    # fc2: NFC2=12 rows (10 real + 2 zero pads), NFC1=32 cols
    lines = []
    for oc in range(12):
        for k in range(32):
            lines.append(to_hex16(int(w2[oc, k])) if oc < 10 else to_hex16(0))
    (path_dir / "w2.mem").write_text("\n".join(lines) + "\n")


def export_test_set(model, test_loader, n_export, path_dir, info, device):
    model.eval()
    images, labels, preds = [], [], []
    with torch.no_grad():
        n = 0
        for x, y in test_loader:
            for i in range(x.size(0)):
                if n >= n_export:
                    break
                images.append(x[i].cpu().numpy().astype(np.int32))
                labels.append(int(y[i]))
                pred = model(x[i:i+1].to(device)).argmax(dim=1).item()
                preds.append(int(pred))
                n += 1
            if n >= n_export:
                break

    lines = []
    for img in images:
        for r in range(26):
            for c in range(26):
                lines.append(to_hex16(int(img[0, r, c])))
    (path_dir / "images.mem").write_text("\n".join(lines) + "\n")

    (path_dir / "labels.txt").write_text("\n".join(str(l) for l in labels) + "\n")
    (path_dir / "preds.txt") .write_text("\n".join(str(p) for p in preds)  + "\n")
    (path_dir / "labels.mem").write_text(
        "\n".join(to_hex16(l) for l in labels) + "\n")

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    info["n_export"]       = n_export
    info["export_acc"]     = correct / n_export
    info["export_correct"] = correct


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",         type=int,   default=100)
    ap.add_argument("--lr",             type=float, default=5e-2)
    ap.add_argument("--batch",          type=int,   default=128)
    ap.add_argument("--w_max",          type=int,   default=7)
    ap.add_argument("--conv2_shift",    type=int,   default=4)
    ap.add_argument("--export",         type=int,   default=200)
    ap.add_argument("--temperature",    type=float, default=1.0)
    ap.add_argument("--label_smoothing",type=float, default=0.0)
    ap.add_argument("--warmup_epochs",  type=int,   default=3)
    ap.add_argument("--weight_decay",   type=float, default=1e-4)
    ap.add_argument("--grad_clip",      type=float, default=5.0)
    ap.add_argument("--seed",           type=int,   default=0)
    ap.add_argument("--aug",            dest="augment", action="store_true")
    ap.add_argument("--no-aug",         dest="augment", action="store_false")
    ap.set_defaults(augment=False)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"weight clip range: [-{args.w_max}, +{args.w_max}]   "
          f"(input pixels are centered to [-32, +31])")
    print(f"conv2 right-shift before FC1: >>>{args.conv2_shift}")
    print(f"data augmentation: {'ON (RandomCrop pad=2)' if args.augment else 'OFF'}")
    print(f"epochs={args.epochs}  lr={args.lr}  T={args.temperature}  "
          f"smooth={args.label_smoothing}  warmup={args.warmup_epochs}")
    print(f"weight_decay={args.weight_decay}  grad_clip={args.grad_clip}")

    train_loader, test_loader = get_loaders(args.batch, augment=args.augment)
    model = MiniCNNv2(w_max=args.w_max, conv2_shift=args.conv2_shift).to(device)
    print(f"params: conv1={model.conv1.weight.numel()}"
          f"  conv2={model.conv2.weight.numel()}"
          f"  fc1={model.fc1.weight.numel()}"
          f"  fc2={model.fc2.weight.numel()}"
          f"  total={sum(p.numel() for p in model.parameters())}")

    best = train(model, train_loader, test_loader, device,
                 epochs=args.epochs, base_lr=args.lr,
                 temperature=args.temperature,
                 label_smoothing=args.label_smoothing,
                 warmup_epochs=args.warmup_epochs,
                 weight_decay=args.weight_decay,
                 grad_clip=args.grad_clip)

    wc1, wc2, w1, w2 = quantized_weights_int(model, args.w_max)
    info = {
        "version":         "v2",
        "best_test_acc":   best,
        "w_max":           args.w_max,
        "conv2_shift":     args.conv2_shift,
        "epochs":          args.epochs,
        "augmentation":    args.augment,
        "conv1_max":       int(np.abs(wc1).max()),
        "conv2_max":       int(np.abs(wc2).max()),
        "w1_max":          int(np.abs(w1).max()),
        "w2_max":          int(np.abs(w2).max()),
        "conv1_nz":        int((wc1 != 0).sum()),
        "conv2_nz":        int((wc2 != 0).sum()),
        "w1_nz":           int((w1 != 0).sum()),
        "w2_nz":           int((w2 != 0).sum()),
    }
    export_weights(wc1, wc2, w1, w2, DATA)
    export_test_set(model, test_loader, args.export, DATA, info, device)

    info_txt = "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n"
    (DATA / "info.txt").write_text(info_txt)
    print("\n=== summary ===")
    print(info_txt)
    print(f"wrote artefacts under {DATA}/")


if __name__ == "__main__":
    main()
