#!/usr/bin/env python3
"""
train_minicnn_cifar10.py
========================
Quantization-aware training + export flow for a CIFAR-10 variant of MiniCNN-v2.

Design goals:
- Keep the export format consistent with existing MNIST scripts.
- Keep weights in INT4-style clipped/rounded values (stored as hex16 words).
- Keep intermediate math aligned to int16/int32 style data flow.

Outputs (under data/cifar10_<arch>/):
    conv*_k.mem  w1.mem  w2.mem                    quantized weights
    images.mem   labels.mem                     N test images + labels
    labels.txt   preds.txt                      human-readable
    info.txt                                    summary metadata
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
CIFAR_DIR = ROOT / "data" / "cifar10"
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class CIFARArch:
    name: str
    input_size: int
    conv1_out: int
    conv2_out: int
    conv3_out: int | None
    pool_kernel: int
    pool_stride: int
    conv_relu: bool
    fc_hidden: int = 32
    conv4_out: int | None = None
    conv5_out: int | None = None
    conv6_out: int | None = None
    conv_padding: int = 0
    v4_vgg_lite: bool = False
    batch_norm: bool = False
    global_avg_pool: bool = False
    single_fc: bool = False

    @property
    def conv1_spatial(self) -> int:
        return self.input_size + 2 * self.conv_padding - 2

    @property
    def pool_spatial(self) -> int:
        return (self.conv1_spatial - self.pool_kernel) // self.pool_stride + 1

    @property
    def conv2_spatial(self) -> int:
        if self.v4_vgg_lite:
            # v4 pools after Conv2, not after Conv1.
            conv2 = self.conv1_spatial + 2 * self.conv_padding - 2
            return (conv2 - self.pool_kernel) // self.pool_stride + 1
        return self.pool_spatial + 2 * self.conv_padding - 2

    @property
    def conv3_spatial(self) -> int:
        if self.conv3_out is None:
            return self.conv2_spatial
        return self.conv2_spatial + 2 * self.conv_padding - 2

    @property
    def conv4_spatial(self) -> int:
        if self.conv4_out is None:
            return self.conv3_spatial
        spatial = self.conv3_spatial + 2 * self.conv_padding - 2
        if self.v4_vgg_lite:
            spatial = (spatial - self.pool_kernel) // self.pool_stride + 1
        return spatial

    @property
    def conv5_spatial(self) -> int:
        if self.conv5_out is None:
            return self.conv4_spatial
        return self.conv4_spatial + 2 * self.conv_padding - 2

    @property
    def conv6_spatial(self) -> int:
        if self.conv6_out is None:
            return self.conv5_spatial
        return self.conv5_spatial + 2 * self.conv_padding - 2

    @property
    def final_conv_out(self) -> int:
        for channels in (self.conv6_out, self.conv5_out, self.conv4_out, self.conv3_out, self.conv2_out):
            if channels is not None:
                return channels
        raise AssertionError("unreachable")

    @property
    def final_conv_spatial(self) -> int:
        if self.conv6_out is not None:
            return self.conv6_spatial
        if self.conv5_out is not None:
            return self.conv5_spatial
        if self.conv4_out is not None:
            return self.conv4_spatial
        if self.conv3_out is not None:
            return self.conv3_spatial
        return self.conv2_spatial

    @property
    def fc_in(self) -> int:
        if self.global_avg_pool:
            return self.final_conv_out
        return self.final_conv_out * self.final_conv_spatial * self.final_conv_spatial

    @property
    def version(self) -> str:
        return f"cifar10_{self.name}"

    @property
    def description(self) -> str:
        if self.v4_vgg_lite:
            if self.batch_norm:
                return (
                    "3x32x32 -> Conv3x3(3->64,pad1)+BN -> ReLU -> "
                    "Conv3x3(64->64,pad1)+BN -> ReLU -> AvgPool2 -> "
                    "Conv3x3(64->128,pad1)+BN -> ReLU -> Conv3x3(128->128,pad1)+BN -> "
                    "ReLU -> AvgPool2 -> Conv3x3(128->128,pad1)+BN -> ReLU -> "
                    "Conv3x3(128->128,pad1)+BN -> ReLU -> GlobalAvgPool8x8 -> FC(128->10)"
                )
            return (
                "3x32x32 -> Conv3x3(3->32,pad1) -> ReLU -> "
                "Conv3x3(32->32,pad1) -> ReLU -> AvgPool2 -> "
                "Conv3x3(32->64,pad1) -> ReLU -> Conv3x3(64->64,pad1) -> "
                "ReLU -> AvgPool2 -> Conv3x3(64->64,pad1) -> ReLU -> "
                "GlobalAvgPool8x8 -> FC(64->10)"
            )
        relu = "ReLU after Conv1/Conv2" if self.conv_relu else "no conv ReLU"
        parts = [
            f"3x{self.input_size}x{self.input_size}",
            f"Conv3x3(3->{self.conv1_out})",
            relu,
            f"AvgPool{self.pool_kernel}",
            f"Conv3x3({self.conv1_out}->{self.conv2_out})",
        ]
        if self.conv3_out is not None:
            parts.extend(
                [
                    "ReLU" if self.conv_relu else "linear",
                    f"Conv3x3({self.conv2_out}->{self.conv3_out})",
                ]
            )
        parts.append(f"FC({self.fc_in}->{self.fc_hidden}->10)")
        return " -> ".join(parts)


ARCHES = {
    "v1": CIFARArch(
        name="v1",
        input_size=26,
        conv1_out=8,
        conv2_out=16,
        conv3_out=None,
        pool_kernel=4,
        pool_stride=4,
        conv_relu=False,
    ),
    # v2 deliberately keeps the 26x26 exported image contract to minimize
    # downstream data-layout disruption while increasing CIFAR model capacity.
    "v2": CIFARArch(
        name="v2",
        input_size=26,
        conv1_out=16,
        conv2_out=32,
        conv3_out=None,
        pool_kernel=2,
        pool_stride=2,
        conv_relu=True,
    ),
    # v3 keeps the v2 3x26x26 exported image contract, but adds a third
    # 3x3/ReLU convolution after Conv2. The extra conv improves CIFAR capacity
    # while reducing FC fan-in from v2's 3200 to 2048.
    "v3": CIFARArch(
        name="v3",
        input_size=26,
        conv1_out=16,
        conv2_out=32,
        conv3_out=32,
        pool_kernel=2,
        pool_stride=2,
        conv_relu=True,
    ),
    # v4 option A from apccasv2.txt: VGG-lite, full 32x32 input, padding-
    # preserving 3x3 convs, AvgPool2 between stages, global average pool, and a
    # single FC classifier. No BN, residual add, or other non-SAGE-mappable op.
    "v4": CIFARArch(
        name="v4",
        input_size=32,
        conv1_out=32,
        conv2_out=32,
        conv3_out=64,
        conv4_out=64,
        conv5_out=64,
        pool_kernel=2,
        pool_stride=2,
        conv_relu=True,
        fc_hidden=10,
        conv_padding=1,
        v4_vgg_lite=True,
        global_avg_pool=True,
        single_fc=True,
    ),
    # v5 option A from apccasv2.txt: wider VGG-BN-style CIFAR model. BatchNorm
    # is used only during float training and folded into conv weights/biases
    # before fixed-point calibration/QAT/export.
    "v5": CIFARArch(
        name="v5",
        input_size=32,
        conv1_out=64,
        conv2_out=64,
        conv3_out=128,
        conv4_out=128,
        conv5_out=128,
        conv6_out=128,
        pool_kernel=2,
        pool_stride=2,
        conv_relu=True,
        fc_hidden=10,
        conv_padding=1,
        v4_vgg_lite=True,
        batch_norm=True,
        global_avg_pool=True,
        single_fc=True,
    ),
    # v6 keeps the v5 VGG-BN topology exactly. It exists as a separate export
    # target so PTQ improvements can be tested without disturbing v5 artifacts.
    "v6": CIFARArch(
        name="v6",
        input_size=32,
        conv1_out=64,
        conv2_out=64,
        conv3_out=128,
        conv4_out=128,
        conv5_out=128,
        conv6_out=128,
        pool_kernel=2,
        pool_stride=2,
        conv_relu=True,
        fc_hidden=10,
        conv_padding=1,
        v4_vgg_lite=True,
        batch_norm=True,
        global_avg_pool=True,
        single_fc=True,
    ),
}


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, g):
        return g


def qw(w: torch.Tensor, w_max: int) -> torch.Tensor:
    return STE.apply(torch.clamp(w, -w_max, w_max))


def nearest_pow2_tensor(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=torch.finfo(x.dtype).tiny)
    return torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), torch.round(torch.log2(x)))


def floor_ste(x: torch.Tensor) -> torch.Tensor:
    return FloorSTE.apply(x)


class MiniCNNCIFAR10(nn.Module):
    """
    CIFAR-10 model with selectable MiniCNN-like geometry.
    """

    def __init__(
        self,
        arch: CIFARArch,
        w_max: int = 6,
        conv2_shift: int = 3,
        v4_pool1_shift: int = 0,
        v4_pool2_shift: int = 0,
        v4_gap_shift: int = 0,
        init_mode: str = "legacy",
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.arch = arch
        self.w_max = w_max
        self.conv2_shift = conv2_shift
        self.v4_pool1_shift = v4_pool1_shift
        self.v4_pool2_shift = v4_pool2_shift
        self.v4_gap_shift = v4_gap_shift
        self.input_scale = input_scale
        self.weight_mode = "quantized"
        self.activation_mode = "integer"
        self.conv1 = nn.Conv2d(3, arch.conv1_out, 3, bias=False)
        self.conv2 = nn.Conv2d(arch.conv1_out, arch.conv2_out, 3, bias=False)
        self.conv3 = (
            nn.Conv2d(arch.conv2_out, arch.conv3_out, 3, padding=arch.conv_padding, bias=False)
            if arch.conv3_out is not None
            else None
        )
        self.conv4 = (
            nn.Conv2d(arch.conv3_out, arch.conv4_out, 3, padding=arch.conv_padding, bias=False)
            if arch.conv4_out is not None and arch.conv3_out is not None
            else None
        )
        self.conv5 = (
            nn.Conv2d(arch.conv4_out, arch.conv5_out, 3, padding=arch.conv_padding, bias=False)
            if arch.conv5_out is not None and arch.conv4_out is not None
            else None
        )
        self.conv6 = (
            nn.Conv2d(arch.conv5_out, arch.conv6_out, 3, padding=arch.conv_padding, bias=False)
            if arch.conv6_out is not None and arch.conv5_out is not None
            else None
        )
        self.bn1 = nn.BatchNorm2d(arch.conv1_out) if arch.batch_norm else None
        self.bn2 = nn.BatchNorm2d(arch.conv2_out) if arch.batch_norm else None
        self.bn3 = nn.BatchNorm2d(arch.conv3_out) if arch.batch_norm and arch.conv3_out is not None else None
        self.bn4 = nn.BatchNorm2d(arch.conv4_out) if arch.batch_norm and arch.conv4_out is not None else None
        self.bn5 = nn.BatchNorm2d(arch.conv5_out) if arch.batch_norm and arch.conv5_out is not None else None
        self.bn6 = nn.BatchNorm2d(arch.conv6_out) if arch.batch_norm and arch.conv6_out is not None else None
        if arch.single_fc:
            self.fc1 = nn.Linear(arch.fc_in, 10, bias=False)
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(arch.fc_in, arch.fc_hidden, bias=False)
            self.fc2 = nn.Linear(arch.fc_hidden, 10, bias=False)
        self.register_buffer("conv1_w_scale", torch.tensor(1.0))
        self.register_buffer("conv2_w_scale", torch.tensor(1.0))
        self.register_buffer("conv3_w_scale", torch.tensor(1.0))
        self.register_buffer("conv4_w_scale", torch.tensor(1.0))
        self.register_buffer("conv5_w_scale", torch.tensor(1.0))
        self.register_buffer("conv6_w_scale", torch.tensor(1.0))
        self.register_buffer("fc1_w_scale", torch.tensor(1.0))
        self.register_buffer("fc2_w_scale", torch.tensor(1.0))
        self.register_buffer("conv1_w_channel_scale", torch.ones(arch.conv1_out) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv2_w_channel_scale", torch.ones(arch.conv2_out) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv3_w_channel_scale", torch.ones(arch.conv3_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv4_w_channel_scale", torch.ones(arch.conv4_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv5_w_channel_scale", torch.ones(arch.conv5_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv6_w_channel_scale", torch.ones(arch.conv6_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv1_w_scale_shift", torch.zeros(arch.conv1_out, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv2_w_scale_shift", torch.zeros(arch.conv2_out, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv3_w_scale_shift", torch.zeros(arch.conv3_out or 0, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv4_w_scale_shift", torch.zeros(arch.conv4_out or 0, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv5_w_scale_shift", torch.zeros(arch.conv5_out or 0, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv6_w_scale_shift", torch.zeros(arch.conv6_out or 0, dtype=torch.int16) if arch.name == "v6" else torch.empty(0, dtype=torch.int16))
        self.register_buffer("conv1_export_b", torch.zeros(arch.conv1_out))
        self.register_buffer("conv2_export_b", torch.zeros(arch.conv2_out))
        self.register_buffer("conv3_export_b", torch.zeros(arch.conv3_out or 0))
        self.register_buffer("conv4_export_b", torch.zeros(arch.conv4_out or 0))
        self.register_buffer("conv5_export_b", torch.zeros(arch.conv5_out or 0))
        self.register_buffer("conv6_export_b", torch.zeros(arch.conv6_out or 0))
        self.register_buffer("conv1_b_scale", torch.tensor(1.0))
        self.register_buffer("conv2_b_scale", torch.tensor(1.0))
        self.register_buffer("conv3_b_scale", torch.tensor(1.0))
        self.register_buffer("conv4_b_scale", torch.tensor(1.0))
        self.register_buffer("conv5_b_scale", torch.tensor(1.0))
        self.register_buffer("conv6_b_scale", torch.tensor(1.0))
        self.register_buffer("conv1_b_channel_scale", torch.ones(arch.conv1_out) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv2_b_channel_scale", torch.ones(arch.conv2_out) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv3_b_channel_scale", torch.ones(arch.conv3_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv4_b_channel_scale", torch.ones(arch.conv4_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv5_b_channel_scale", torch.ones(arch.conv5_out or 0) if arch.name == "v6" else torch.empty(0))
        self.register_buffer("conv6_b_channel_scale", torch.ones(arch.conv6_out or 0) if arch.name == "v6" else torch.empty(0))
        self._init_weights(init_mode)

    def _init_weights(self, init_mode: str) -> None:
        if init_mode == "legacy":
            layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.fc1, self.fc2]
            weights = [layer.weight for layer in layers if layer is not None]
            for p in weights:
                nn.init.normal_(p, mean=0.0, std=0.6)
            return
        if init_mode != "kaiming":
            raise ValueError(f"unknown init_mode: {init_mode}")

        # v2 sees raw centered integer pixels. Scale only the first layer init so
        # early logits stay in a trainable range without changing export inputs.
        raw_pixel_std = 18.5
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.conv1.weight.data.div_(raw_pixel_std)
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        if self.conv3 is not None:
            nn.init.kaiming_normal_(self.conv3.weight, mode="fan_in", nonlinearity="relu")
        if self.conv4 is not None:
            nn.init.kaiming_normal_(self.conv4.weight, mode="fan_in", nonlinearity="relu")
        if self.conv5 is not None:
            nn.init.kaiming_normal_(self.conv5.weight, mode="fan_in", nonlinearity="relu")
        if self.conv6 is not None:
            nn.init.kaiming_normal_(self.conv6.weight, mode="fan_in", nonlinearity="relu")
        if self.fc2 is None:
            nn.init.xavier_normal_(self.fc1.weight)
        else:
            nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
            nn.init.xavier_normal_(self.fc2.weight)

    def set_modes(self, weight_mode: str, activation_mode: str) -> None:
        if weight_mode not in {"float", "quantized"}:
            raise ValueError(f"unknown weight_mode: {weight_mode}")
        if activation_mode not in {"float", "integer"}:
            raise ValueError(f"unknown activation_mode: {activation_mode}")
        self.weight_mode = weight_mode
        self.activation_mode = activation_mode

    def _weight_multiplier(self, name: str, p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        channel_scale = getattr(self, f"{name}_w_channel_scale", None)
        if channel_scale is not None and channel_scale.numel() > 0:
            return channel_scale.to(device=p.device, dtype=p.dtype).reshape(-1, 1, 1, 1)
        return scale

    def _weight(self, name: str, p: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "quantized":
            return qw(p * self._weight_multiplier(name, p, scale), self.w_max)
        return torch.clamp(p, -self.w_max, self.w_max)

    def _conv_bias(self, name: str) -> torch.Tensor | None:
        bias = getattr(self, f"{name}_export_b")
        if bias.numel() == 0:
            return None
        channel_scale = getattr(self, f"{name}_b_channel_scale", None)
        if self.weight_mode == "quantized" and channel_scale is not None and channel_scale.numel() > 0:
            scale = channel_scale.to(device=bias.device, dtype=bias.dtype)
            return STE.apply(torch.clamp(bias * scale, INT32_MIN, INT32_MAX))
        if self.weight_mode == "quantized":
            return STE.apply(torch.clamp(bias * getattr(self, f"{name}_b_scale"), INT32_MIN, INT32_MAX))
        return bias

    def _maybe_bn(self, h: torch.Tensor, bn: nn.BatchNorm2d | None) -> torch.Tensor:
        if bn is not None and self.weight_mode == "float":
            return bn(h)
        return h

    def _activation_shift(self, h: torch.Tensor, shift: int) -> torch.Tensor:
        if shift <= 0:
            return h
        h = h / float(1 << shift)
        if self.activation_mode == "integer":
            h = floor_ste(h)
        return h

    def _v6_weight_rescale(self, h: torch.Tensor, name: str) -> torch.Tensor:
        scale_shift = getattr(self, f"{name}_w_scale_shift", None)
        if self.arch.name != "v6" or self.weight_mode != "quantized" or scale_shift is None or scale_shift.numel() == 0:
            return h
        shifts = scale_shift.to(device=h.device, dtype=torch.int64).view(1, -1, 1, 1)
        pos = torch.clamp(shifts, min=0)
        neg = torch.clamp(-shifts, min=0)
        h = h * torch.pow(torch.tensor(2.0, device=h.device, dtype=h.dtype), pos.to(h.dtype))
        h = h / torch.pow(torch.tensor(2.0, device=h.device, dtype=h.dtype), neg.to(h.dtype))
        if self.activation_mode == "integer":
            h = floor_ste(h)
        return h

    def forward(self, x_int: torch.Tensor) -> torch.Tensor:
        wc1 = self._weight("conv1", self.conv1.weight, self.conv1_w_scale)
        wc2 = self._weight("conv2", self.conv2.weight, self.conv2_w_scale)
        wc3 = (
            self._weight("conv3", self.conv3.weight, self.conv3_w_scale)
            if self.conv3 is not None
            else None
        )
        wc4 = (
            self._weight("conv4", self.conv4.weight, self.conv4_w_scale)
            if self.conv4 is not None
            else None
        )
        wc5 = (
            self._weight("conv5", self.conv5.weight, self.conv5_w_scale)
            if self.conv5 is not None
            else None
        )
        wc6 = (
            self._weight("conv6", self.conv6.weight, self.conv6_w_scale)
            if self.conv6 is not None
            else None
        )
        w1 = self._weight("fc1", self.fc1.weight, self.fc1_w_scale)
        w2 = self._weight("fc2", self.fc2.weight, self.fc2_w_scale) if self.fc2 is not None else None

        if self.activation_mode == "float" and self.input_scale != 1.0:
            x_int = x_int / self.input_scale
        h = F.conv2d(x_int, wc1, bias=self._conv_bias("conv1"), padding=self.arch.conv_padding)
        h = self._v6_weight_rescale(h, "conv1")
        h = self._maybe_bn(h, self.bn1)
        if self.arch.conv_relu:
            h = F.relu(h)
        if not self.arch.v4_vgg_lite:
            h = F.avg_pool2d(h, self.arch.pool_kernel, self.arch.pool_stride)
            if self.activation_mode == "integer":
                h = floor_ste(h)
        h = F.conv2d(h, wc2, bias=self._conv_bias("conv2"), padding=self.arch.conv_padding)
        h = self._v6_weight_rescale(h, "conv2")
        h = self._maybe_bn(h, self.bn2)
        if self.arch.conv_relu:
            h = F.relu(h)
        if self.arch.v4_vgg_lite:
            h = F.avg_pool2d(h, self.arch.pool_kernel, self.arch.pool_stride)
            if self.activation_mode == "integer":
                h = floor_ste(h)
            h = self._activation_shift(h, self.v4_pool1_shift)
        if wc3 is not None:
            h = F.conv2d(h, wc3, bias=self._conv_bias("conv3"), padding=self.arch.conv_padding)
            h = self._v6_weight_rescale(h, "conv3")
            h = self._maybe_bn(h, self.bn3)
            if self.arch.conv_relu:
                h = F.relu(h)
        if wc4 is not None:
            h = F.conv2d(h, wc4, bias=self._conv_bias("conv4"), padding=self.arch.conv_padding)
            h = self._v6_weight_rescale(h, "conv4")
            h = self._maybe_bn(h, self.bn4)
            if self.arch.conv_relu:
                h = F.relu(h)
            if self.arch.v4_vgg_lite:
                h = F.avg_pool2d(h, self.arch.pool_kernel, self.arch.pool_stride)
                if self.activation_mode == "integer":
                    h = floor_ste(h)
                h = self._activation_shift(h, self.v4_pool2_shift)
        if wc5 is not None:
            h = F.conv2d(h, wc5, bias=self._conv_bias("conv5"), padding=self.arch.conv_padding)
            h = self._v6_weight_rescale(h, "conv5")
            h = self._maybe_bn(h, self.bn5)
            if self.arch.conv_relu:
                h = F.relu(h)
        if wc6 is not None:
            h = F.conv2d(h, wc6, bias=self._conv_bias("conv6"), padding=self.arch.conv_padding)
            h = self._v6_weight_rescale(h, "conv6")
            h = self._maybe_bn(h, self.bn6)
            if self.arch.conv_relu:
                h = F.relu(h)
        if self.arch.global_avg_pool:
            h = F.avg_pool2d(h, h.shape[-1])
            if self.activation_mode == "integer":
                h = floor_ste(h)
            h = self._activation_shift(h, self.v4_gap_shift)
        h = self._activation_shift(h, self.conv2_shift)
        h = h.flatten(1)
        h = F.linear(h, w1)
        if w2 is not None:
            h = F.relu(h)
            h = F.linear(h, w2)
        return h


def _maybe_subset(ds, n: int, seed: int):
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def get_loaders(
    batch: int,
    arch: CIFARArch,
    augment: bool,
    subset_train: int,
    subset_test: int,
    synthetic: bool,
    seed: int,
):
    if synthetic:
        n_train = subset_train if subset_train > 0 else 512
        n_test = subset_test if subset_test > 0 else 256
        train_ds = datasets.FakeData(
            size=n_train,
            image_size=(3, arch.input_size, arch.input_size),
            num_classes=10,
            transform=transforms.PILToTensor(),
            random_offset=seed,
        )
        test_ds = datasets.FakeData(
            size=n_test,
            image_size=(3, arch.input_size, arch.input_size),
            num_classes=10,
            transform=transforms.PILToTensor(),
            random_offset=seed + 1,
        )
    else:
        train_tf_list = [transforms.CenterCrop(arch.input_size)]
        if augment:
            train_tf_list.extend(
                [
                    transforms.RandomCrop(
                        arch.input_size,
                        padding=4 if arch.input_size == 32 else 2,
                        padding_mode="constant",
                        fill=0,
                    ),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        train_tf_list.append(transforms.PILToTensor())
        train_tf = transforms.Compose(train_tf_list)
        test_tf = transforms.Compose([transforms.CenterCrop(arch.input_size), transforms.PILToTensor()])

        train_ds = datasets.CIFAR10(str(CIFAR_DIR), train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(str(CIFAR_DIR), train=False, download=True, transform=test_tf)
        train_ds = _maybe_subset(train_ds, subset_train, seed)
        test_ds = _maybe_subset(test_ds, subset_test, seed + 1)

    def to_centered_int(batch_items):
        x = torch.stack([img for img, _ in batch_items]).float()
        x = (x / 4.0).floor() - 32.0
        y = torch.tensor([label for _, label in batch_items], dtype=torch.long)
        return x, y

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=0, collate_fn=to_centered_int
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch, shuffle=False, num_workers=0, collate_fn=to_centered_int
    )
    return train_loader, test_loader, len(train_ds), len(test_ds)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def evaluate_max_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> float:
    if max_samples <= 0:
        return evaluate(model, loader, device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            remaining = max_samples - total
            if remaining <= 0:
                break
            if x.size(0) > remaining:
                x = x[:remaining]
                y = y[:remaining]
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def evaluate_modes(
    model: MiniCNNCIFAR10,
    loader: DataLoader,
    device: torch.device,
    weight_mode: str,
    activation_mode: str,
    max_samples: int = 0,
) -> float:
    old_weight_mode = model.weight_mode
    old_activation_mode = model.activation_mode
    model.set_modes(weight_mode, activation_mode)
    try:
        return evaluate_max_samples(model, loader, device, max_samples)
    finally:
        model.set_modes(old_weight_mode, old_activation_mode)


def set_activation_shifts(
    model: MiniCNNCIFAR10,
    conv2_shift: int,
    v4_pool1_shift: int | None = None,
    v4_pool2_shift: int | None = None,
    v4_gap_shift: int | None = None,
) -> None:
    model.conv2_shift = conv2_shift
    if v4_pool1_shift is not None:
        model.v4_pool1_shift = v4_pool1_shift
    if v4_pool2_shift is not None:
        model.v4_pool2_shift = v4_pool2_shift
    if v4_gap_shift is not None:
        model.v4_gap_shift = v4_gap_shift


def fold_batch_norm_for_export(model: MiniCNNCIFAR10) -> bool:
    if not model.arch.batch_norm:
        return False

    already_folded = True
    has_folded_bias = False
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
        conv = getattr(model, name)
        bn = getattr(model, f"bn{name[-1]}")
        if conv is None or bn is None:
            continue
        bias = getattr(model, f"{name}_export_b")
        has_folded_bias = has_folded_bias or (bias.numel() > 0 and bool(torch.any(bias != 0).item()))
        already_folded = already_folded and bool(torch.allclose(bn.weight.detach(), torch.ones_like(bn.weight)))
        already_folded = already_folded and bool(torch.allclose(bn.bias.detach(), torch.zeros_like(bn.bias)))
        already_folded = already_folded and bool(torch.allclose(bn.running_mean.detach(), torch.zeros_like(bn.running_mean)))
        already_folded = already_folded and bool(torch.allclose(bn.running_var.detach(), torch.ones_like(bn.running_var)))
    if already_folded and has_folded_bias:
        return True

    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
        conv = getattr(model, name)
        bn = getattr(model, f"bn{name[-1]}")
        if conv is None or bn is None:
            continue

        scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
        folded_weight = conv.weight.detach() * scale.reshape(-1, 1, 1, 1)
        folded_bias = bn.bias.detach() - scale * bn.running_mean.detach()
        conv.weight.data.copy_(folded_weight)
        getattr(model, f"{name}_export_b").data.copy_(folded_bias)

        # Keep BN modules neutral so accidental float-mode evaluation remains
        # equivalent to the folded graph.
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
    return True


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) not in {2, 3}:
                raise ValueError(f"bad integer range: {part}")
            start, stop = bits[0], bits[1]
            step = bits[2] if len(bits) == 3 else 1
            if step <= 0:
                raise ValueError(f"range step must be positive: {part}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("empty integer list")
    return sorted(set(values))


def sweep_activation_shifts(
    model: MiniCNNCIFAR10,
    loader: DataLoader,
    device: torch.device,
    conv2_values: list[int],
    pool1_values: list[int],
    pool2_values: list[int],
    gap_values: list[int],
    max_samples: int = 0,
    topk: int = 8,
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    if not model.arch.v4_vgg_lite:
        raise ValueError("--shift-sweep is only supported for VGG-style CIFAR architectures")

    old = (model.conv2_shift, model.v4_pool1_shift, model.v4_pool2_shift, model.v4_gap_shift)
    results: list[dict[str, int | float]] = []
    best: dict[str, int | float] | None = None
    try:
        for pool1_shift in pool1_values:
            for pool2_shift in pool2_values:
                for gap_shift in gap_values:
                    for conv2_shift in conv2_values:
                        set_activation_shifts(model, conv2_shift, pool1_shift, pool2_shift, gap_shift)
                        acc = evaluate_modes(
                            model,
                            loader,
                            device,
                            "quantized",
                            "integer",
                            max_samples=max_samples,
                        )
                        item: dict[str, int | float] = {
                            "conv2_shift": conv2_shift,
                            "v4_pool1_shift": pool1_shift,
                            "v4_pool2_shift": pool2_shift,
                            "v4_gap_shift": gap_shift,
                            "q_acc": acc,
                        }
                        results.append(item)
                        if best is None or acc > float(best["q_acc"]):
                            best = item
        assert best is not None
        ranked = sorted(results, key=lambda x: float(x["q_acc"]), reverse=True)
        return best, ranked[:topk]
    finally:
        set_activation_shifts(model, *old)


def warmup_cosine_lr(opt, epoch: int, epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def train(
    model: MiniCNNCIFAR10,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    base_lr: float,
    temperature: float,
    label_smoothing: float,
    warmup_epochs: int,
    weight_decay: float,
    grad_clip: float,
    weight_mode: str,
    activation_mode: str,
    phase_name: str,
    opt_name: str,
    select_weight_mode: str = "quantized",
    select_activation_mode: str = "integer",
    report_float_acc: bool = False,
) -> float:
    if epochs <= 0:
        return evaluate_modes(model, test_loader, device, select_weight_mode, select_activation_mode)

    model.set_modes(weight_mode, activation_mode)
    if opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"unknown optimizer: {opt_name}")
    best_acc = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print(
        f"\n=== {phase_name}: train={weight_mode}/{activation_mode}, "
        f"select={select_weight_mode}/{select_activation_mode} ==="
    )
    for ep in range(epochs):
        lr = warmup_cosine_lr(opt, ep, epochs, warmup_epochs, base_lr)
        model.set_modes(weight_mode, activation_mode)
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out / temperature, y, label_smoothing=label_smoothing)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()

        float_acc = (
            evaluate_modes(model, test_loader, device, "float", "float")
            if report_float_acc
            else None
        )
        acc = evaluate_modes(model, test_loader, device, select_weight_mode, select_activation_mode)
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        if select_weight_mode == "float" and select_activation_mode == "float":
            acc_label = "float_acc"
        elif select_weight_mode == "quantized" and select_activation_mode == "integer":
            acc_label = "q_acc"
        else:
            acc_label = f"{select_weight_mode}_{select_activation_mode}_acc"
        float_part = f"float_acc={float_acc*100:.2f}%  " if float_acc is not None else ""
        print(
            f"  {phase_name} ep {ep+1:3d}/{epochs}  loss={running/max(n,1):.4f}  "
            f"{float_part}{acc_label}={acc*100:.2f}%  lr={lr:.4f}{marker}"
        )

    model.load_state_dict(best_state)
    model.set_modes(select_weight_mode, select_activation_mode)
    return best_acc


def format_pow2_scale(shift: int) -> str:
    return "1" if shift == 0 else f"1/2^{shift}"


def to_hex16(v: int) -> str:
    return f"{v & 0xFFFF:04x}"


def to_hex32(v: int) -> str:
    return f"{v & 0xFFFFFFFF:08x}"


def _weight_multiplier_values(model: MiniCNNCIFAR10, name: str) -> torch.Tensor:
    channel_scale = getattr(model, f"{name}_w_channel_scale", None)
    if channel_scale is not None and channel_scale.numel() > 0:
        return channel_scale.detach()
    return getattr(model, f"{name}_w_scale").detach().reshape(1)


def _mean_weight_multiplier(model: MiniCNNCIFAR10, name: str) -> float:
    return float(_weight_multiplier_values(model, name).float().mean().item())


def calibrate_weight_scales(model: MiniCNNCIFAR10, w_max: int) -> dict[str, float]:
    """
    Fold per-layer post-training scales into exported integer weights.

    Float training leaves many useful CIFAR weights below one integer LSB. Scaling each
    layer up before rounding preserves the existing .mem layout while avoiding an
    all-near-zero export path.
    """
    scales: dict[str, float] = {}
    for name, layer in (
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("conv3", model.conv3),
        ("conv4", model.conv4),
        ("conv5", model.conv5),
        ("conv6", model.conv6),
        ("fc1", model.fc1),
        ("fc2", model.fc2),
    ):
        if layer is None:
            continue
        channel_scale = getattr(model, f"{name}_w_channel_scale", None)
        scale_shift = getattr(model, f"{name}_w_scale_shift", None)
        if model.arch.name == "v6" and name.startswith("conv") and channel_scale is not None and channel_scale.numel() > 0:
            weight = layer.weight.detach()
            max_abs_ch = weight.abs().flatten(1).max(dim=1).values
            step_ch = torch.where(
                max_abs_ch <= 0.0,
                torch.ones_like(max_abs_ch),
                nearest_pow2_tensor(max_abs_ch / float(w_max)),
            )
            multiplier_ch = 1.0 / step_ch
            shift_ch = torch.round(torch.log2(step_ch)).to(torch.int16)
            channel_scale.copy_(multiplier_ch.to(channel_scale.device, channel_scale.dtype))
            scale_shift.copy_(shift_ch.to(scale_shift.device))
            getattr(model, f"{name}_w_scale").fill_(1.0)
            scales[f"{name}_w_scale_min"] = float(multiplier_ch.min().item())
            scales[f"{name}_w_scale_max"] = float(multiplier_ch.max().item())
            scales[f"{name}_w_scale_shift_min"] = float(shift_ch.min().item())
            scales[f"{name}_w_scale_shift_max"] = float(shift_ch.max().item())
            continue
        max_abs = float(layer.weight.detach().abs().max().item())
        scale = 1.0 if max_abs <= 0.0 else float(w_max) / max_abs
        getattr(model, f"{name}_w_scale").fill_(scale)
        scales[f"{name}_w_scale"] = scale
    return scales


def _vgg_conv_bias_input_scales(model: MiniCNNCIFAR10) -> dict[str, float]:
    """
    Scale folded BN biases into the same accumulator units as quantized conv sums.

    A conv output uses quantized input activations and quantized weights. For v5,
    the input activation entering each folded conv already includes all previous
    weight scales and hardware right-shifts, so bias quantization must include the
    cumulative input scale as well as the current layer's weight scale.
    """
    scale = 1.0
    scales = {"conv1": scale}

    if model.arch.name != "v6":
        scale *= _mean_weight_multiplier(model, "conv1")
    scales["conv2"] = scale
    if model.arch.name != "v6":
        scale *= _mean_weight_multiplier(model, "conv2")
    scale /= float(1 << model.v4_pool1_shift)

    if model.conv3 is not None:
        scales["conv3"] = scale
        if model.arch.name != "v6":
            scale *= _mean_weight_multiplier(model, "conv3")
    if model.conv4 is not None:
        scales["conv4"] = scale
        if model.arch.name != "v6":
            scale *= _mean_weight_multiplier(model, "conv4")
        scale /= float(1 << model.v4_pool2_shift)
    if model.conv5 is not None:
        scales["conv5"] = scale
        if model.arch.name != "v6":
            scale *= _mean_weight_multiplier(model, "conv5")
    if model.conv6 is not None:
        scales["conv6"] = scale
    return scales


def calibrate_conv_bias_scales(model: MiniCNNCIFAR10) -> dict[str, float]:
    scales: dict[str, float] = {}
    if not model.arch.batch_norm:
        for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
            getattr(model, f"{name}_b_scale").fill_(float(getattr(model, f"{name}_w_scale").item()))
        return scales

    input_scales = _vgg_conv_bias_input_scales(model)
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
        bias = getattr(model, f"{name}_export_b")
        if bias.numel() == 0:
            continue
        channel_scale = getattr(model, f"{name}_b_channel_scale", None)
        weight_scale_values = _weight_multiplier_values(model, name)
        if channel_scale is not None and channel_scale.numel() > 0:
            scale_values = input_scales[name] * weight_scale_values.to(channel_scale.device, channel_scale.dtype)
            channel_scale.copy_(scale_values)
            scales[f"{name}_b_scale_min"] = float(scale_values.min().item())
            scales[f"{name}_b_scale_max"] = float(scale_values.max().item())
            continue
        scale = input_scales[name] * float(weight_scale_values.item())
        getattr(model, f"{name}_b_scale").fill_(scale)
        scales[f"{name}_b_scale"] = scale
    return scales


def quantized_weights_int(model: MiniCNNCIFAR10, w_max: int):
    def q(name, p, scale):
        multiplier = model._weight_multiplier(name, p, scale)
        return torch.round(torch.clamp((p.detach() * multiplier).detach(), -w_max, w_max)).int().cpu().numpy()

    return (
        q("conv1", model.conv1.weight, model.conv1_w_scale),
        q("conv2", model.conv2.weight, model.conv2_w_scale),
        q("conv3", model.conv3.weight, model.conv3_w_scale) if model.conv3 is not None else None,
        q("conv4", model.conv4.weight, model.conv4_w_scale) if model.conv4 is not None else None,
        q("conv5", model.conv5.weight, model.conv5_w_scale) if model.conv5 is not None else None,
        q("conv6", model.conv6.weight, model.conv6_w_scale) if model.conv6 is not None else None,
        q("fc1", model.fc1.weight, model.fc1_w_scale),
        q("fc2", model.fc2.weight, model.fc2_w_scale) if model.fc2 is not None else None,
    )


def export_conv_weight(weight, path: Path) -> None:
    lines = []
    for oc in range(weight.shape[0]):
        for ic in range(weight.shape[1]):
            for i in range(weight.shape[2]):
                for j in range(weight.shape[3]):
                    lines.append(to_hex16(int(weight[oc, ic, i, j])))
    path.write_text("\n".join(lines) + "\n")


def export_vector(values, path: Path) -> None:
    lines = [to_hex16(int(v)) for v in np.asarray(values).reshape(-1)]
    path.write_text("\n".join(lines) + "\n")


def export_vector32(values, path: Path) -> None:
    lines = [to_hex32(int(v)) for v in np.asarray(values).reshape(-1)]
    path.write_text("\n".join(lines) + "\n")


def export_v6_scale_shifts(model: MiniCNNCIFAR10, path_dir: Path) -> None:
    if model.arch.name != "v6":
        return
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
        scale_shift = getattr(model, f"{name}_w_scale_shift")
        if scale_shift.numel() > 0:
            export_vector(scale_shift.cpu().numpy(), path_dir / f"{name}_w_scale_shift.mem")


def export_weights(wc1, wc2, wc3, wc4, wc5, wc6, w1, w2, model: MiniCNNCIFAR10, path_dir: Path) -> None:
    export_conv_weight(wc1, path_dir / "conv1_k.mem")
    export_conv_weight(wc2, path_dir / "conv2_k.mem")
    if wc3 is not None:
        export_conv_weight(wc3, path_dir / "conv3_k.mem")
    if wc4 is not None:
        export_conv_weight(wc4, path_dir / "conv4_k.mem")
    if wc5 is not None:
        export_conv_weight(wc5, path_dir / "conv5_k.mem")
    if wc6 is not None:
        export_conv_weight(wc6, path_dir / "conv6_k.mem")

    for name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv6"):
        bias = getattr(model, f"{name}_export_b")
        if bias.numel() > 0 and torch.any(bias != 0):
            channel_scale = getattr(model, f"{name}_b_channel_scale", None)
            scale = channel_scale if channel_scale is not None and channel_scale.numel() > 0 else getattr(model, f"{name}_b_scale")
            q_bias = (
                torch.round(torch.clamp((bias.detach() * scale).detach(), INT32_MIN, INT32_MAX))
                .to(torch.int64)
                .cpu()
                .numpy()
            )
            export_vector32(q_bias, path_dir / f"{name}_b.mem")
    export_v6_scale_shifts(model, path_dir)

    lines = []
    for oc in range(w1.shape[0]):
        for k in range(w1.shape[1]):
            lines.append(to_hex16(int(w1[oc, k])))
    (path_dir / "w1.mem").write_text("\n".join(lines) + "\n")

    if w2 is not None:
        lines = []
        for oc in range(12):
            for k in range(w2.shape[1]):
                lines.append(to_hex16(int(w2[oc, k])) if oc < w2.shape[0] else to_hex16(0))
        (path_dir / "w2.mem").write_text("\n".join(lines) + "\n")


def export_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    n_export: int,
    path_dir: Path,
    info: dict,
    device: torch.device,
    arch: CIFARArch,
) -> None:
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
                pred = model(x[i : i + 1].to(device)).argmax(dim=1).item()
                preds.append(int(pred))
                n += 1
            if n >= n_export:
                break

    lines = []
    for img in images:
        for ch in range(3):
            for r in range(arch.input_size):
                for c in range(arch.input_size):
                    lines.append(to_hex16(int(img[ch, r, c])))
    (path_dir / "images.mem").write_text("\n".join(lines) + "\n")

    (path_dir / "labels.txt").write_text("\n".join(str(v) for v in labels) + "\n")
    (path_dir / "preds.txt").write_text("\n".join(str(v) for v in preds) + "\n")
    (path_dir / "labels.mem").write_text("\n".join(to_hex16(v) for v in labels) + "\n")

    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    n_actual = len(labels)
    info["n_export_requested"] = n_export
    info["n_export_actual"] = n_actual
    info["export_acc"] = correct / max(n_actual, 1)
    info["export_correct"] = correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHES), default="v1", help="Model/export architecture")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artifact directory; default is data/cifar10_<arch>",
    )
    ap.add_argument("--epochs", type=int, default=100, help="Total epochs across all stages")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--w_max", type=int, default=6)
    ap.add_argument("--conv2_shift", type=int, default=3)
    ap.add_argument("--v4-pool1-shift", type=int, default=0, help="VGG-style shift after first AvgPool2")
    ap.add_argument("--v4-pool2-shift", type=int, default=0, help="VGG-style shift after second AvgPool2")
    ap.add_argument("--v4-gap-shift", type=int, default=0, help="VGG-style shift after GlobalAvgPool")
    ap.add_argument("--shift-sweep", action="store_true", help="VGG-style: sweep activation shifts after weight calibration")
    ap.add_argument("--sweep-conv2", default="0:8", help="comma/range list for --shift-sweep conv2_shift")
    ap.add_argument("--sweep-pool1", default="0:3", help="comma/range list for --shift-sweep v4 pool1 shift")
    ap.add_argument("--sweep-pool2", default="0:3", help="comma/range list for --shift-sweep v4 pool2 shift")
    ap.add_argument("--sweep-gap", default="0:4", help="comma/range list for --shift-sweep v4 global-pool shift")
    ap.add_argument("--sweep-samples", type=int, default=2000, help="test samples used per shift candidate; 0 means full test set")
    ap.add_argument("--qat", action="store_true", help="VGG-style: run fixed-point fake-quant fine-tuning after calibration")
    ap.add_argument(
        "--qat-epochs",
        type=int,
        default=10,
        help="QAT fine-tune epochs when --qat is enabled; independent of --quant-epochs",
    )
    ap.add_argument(
        "--qat-lr-scale",
        type=float,
        default=0.05,
        help="QAT learning-rate multiplier relative to --lr",
    )
    ap.add_argument(
        "--act-scale-mode",
        choices=("fixed", "sweep"),
        default="sweep",
        help="QAT activation scale mode: fixed uses provided shifts, sweep selects integer shifts first",
    )
    ap.add_argument("--save-checkpoint", type=Path, default=None, help="Save best calibrated/export model checkpoint")
    ap.add_argument("--load-checkpoint", type=Path, default=None, help="Load model checkpoint before calibration/sweep/export")
    ap.add_argument(
        "--init",
        choices=("auto", "legacy", "kaiming"),
        default="auto",
        help="Weight initialization; auto keeps v1 legacy init and uses Kaiming for v2/v3/v4/v5/v6",
    )
    ap.add_argument(
        "--opt",
        choices=("auto", "sgd", "adamw"),
        default="auto",
        help="Optimizer; auto keeps v1 SGD and uses AdamW for v2/v3/v4 and SGD for v5/v6",
    )
    ap.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Float-activation diagnostic scale for centered integer inputs; export/integer path stays unscaled",
    )
    ap.add_argument("--export", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument(
        "--relaxed-epochs",
        type=int,
        default=None,
        help="Float-clamped activation pretraining epochs; default is 75%% of --epochs",
    )
    ap.add_argument(
        "--quant-epochs",
        type=int,
        default=None,
        help="Exact quantized/integer fine-tune epochs; default is remaining --epochs",
    )
    ap.add_argument(
        "--no-relaxed",
        action="store_true",
        help="Train only the exact quantized/integer path, matching the old behavior",
    )
    ap.add_argument(
        "--float-only",
        action="store_true",
        help="Diagnostic: train/evaluate float weights and activations only; skip quantized export",
    )
    ap.add_argument("--subset-train", type=int, default=0, help="0 means full train split")
    ap.add_argument("--subset-test", type=int, default=0, help="0 means full test split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aug", dest="augment", action="store_true")
    ap.add_argument("--no-aug", dest="augment", action="store_false")
    ap.set_defaults(augment=True)
    ap.add_argument("--synthetic", action="store_true", help="Use torchvision FakeData (no download)")
    ap.add_argument("--smoke", action="store_true", help="Fast sanity run")
    args = ap.parse_args()
    if args.qat and not ARCHES[args.arch].v4_vgg_lite:
        raise ValueError("--qat is currently supported only for VGG-style CIFAR architectures")
    if args.qat and args.float_only:
        raise ValueError("--qat cannot be combined with --float-only")
    if args.qat and args.qat_epochs <= 0:
        raise ValueError("--qat-epochs must be positive when --qat is enabled")
    if args.qat and args.act_scale_mode == "sweep":
        args.shift_sweep = True
    arch = ARCHES[args.arch]
    data_dir = args.out_dir if args.out_dir is not None else ROOT / "data" / arch.version
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.relaxed_epochs = 1
        if args.quant_epochs is None:
            args.quant_epochs = max(1, args.epochs - args.relaxed_epochs)
        else:
            args.quant_epochs = max(0, min(args.quant_epochs, args.epochs - args.relaxed_epochs))
        args.subset_train = args.subset_train if args.subset_train > 0 else 1024
        args.subset_test = args.subset_test if args.subset_test > 0 else 256
        args.export = min(args.export, 16)
        args.qat_epochs = min(args.qat_epochs, 1)
        if args.qat and args.act_scale_mode == "sweep":
            args.sweep_conv2 = "0,3"
            args.sweep_pool1 = "0,1"
            args.sweep_pool2 = "0,1"
            args.sweep_gap = "0,1"
            args.sweep_samples = min(args.sweep_samples, 64)
        args.augment = False

    if args.float_only:
        relaxed_epochs = args.epochs
        quant_epochs = 0
    elif args.no_relaxed:
        relaxed_epochs = 0
        quant_epochs = args.quant_epochs if args.quant_epochs is not None else args.epochs
    else:
        relaxed_epochs = args.relaxed_epochs
        if relaxed_epochs is None:
            relaxed_epochs = int(round(args.epochs * 0.75))
        relaxed_epochs = max(0, min(relaxed_epochs, args.epochs))
        quant_epochs = args.quant_epochs
        if quant_epochs is None:
            quant_epochs = max(0, args.epochs - relaxed_epochs)
    total_epochs = relaxed_epochs + quant_epochs
    init_mode = args.init
    if init_mode == "auto":
        init_mode = "kaiming" if args.arch in {"v2", "v3", "v4", "v5", "v6"} else "legacy"
    opt_name = args.opt
    if opt_name == "auto":
        opt_name = "sgd" if args.arch in {"v5", "v6"} else ("adamw" if args.arch in {"v2", "v3", "v4"} else "sgd")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"arch: {args.arch} ({arch.version})")
    print(f"topology: {arch.description}")
    print(f"data source: {'synthetic/fakedata' if args.synthetic else 'cifar10'}")
    print(f"weight clip range: [-{args.w_max}, +{args.w_max}]")
    print(f"input centering: floor(u8/4)-32  (range [-32,+31])")
    print(f"post-conv/global-pool right-shift before classifier: >>>{args.conv2_shift}")
    if arch.v4_vgg_lite:
        print(
            "VGG activation shifts: "
            f"pool1>>>{args.v4_pool1_shift} pool2>>>{args.v4_pool2_shift} "
            f"gap>>>{args.v4_gap_shift} final>>>{args.conv2_shift}"
        )
    if arch.batch_norm:
        print("batch norm: training-time BN will be folded before fixed-point export")
    print(f"augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"initialization: {init_mode}")
    print(f"optimizer: {opt_name}")
    print(f"float input scale: {args.input_scale}")
    print(f"training mode: {'float-only diagnostic' if args.float_only else 'staged/export'}")
    print(f"training stages: relaxed={relaxed_epochs} quantized={quant_epochs}")
    if args.qat:
        print(
            "VGG QAT: "
            f"enabled qat_epochs={args.qat_epochs} act_scale_mode={args.act_scale_mode} "
            f"qat_lr={args.lr * args.qat_lr_scale:g}"
        )
    print(
        f"epochs={total_epochs} lr={args.lr} batch={args.batch} export={args.export} "
        f"subset_train={args.subset_train} subset_test={args.subset_test}"
    )

    train_loader, test_loader, n_train, n_test = get_loaders(
        args.batch,
        arch,
        args.augment,
        args.subset_train,
        args.subset_test,
        args.synthetic,
        args.seed,
    )
    model = MiniCNNCIFAR10(
        arch=arch,
        w_max=args.w_max,
        conv2_shift=args.conv2_shift,
        v4_pool1_shift=args.v4_pool1_shift,
        v4_pool2_shift=args.v4_pool2_shift,
        v4_gap_shift=args.v4_gap_shift,
        init_mode=init_mode,
        input_scale=args.input_scale,
    ).to(device)
    print(
        f"params: conv1={model.conv1.weight.numel()} conv2={model.conv2.weight.numel()} "
        f"conv3={model.conv3.weight.numel() if model.conv3 is not None else 0} "
        f"conv4={model.conv4.weight.numel() if model.conv4 is not None else 0} "
        f"conv5={model.conv5.weight.numel() if model.conv5 is not None else 0} "
        f"conv6={model.conv6.weight.numel() if model.conv6 is not None else 0} "
        f"fc1={model.fc1.weight.numel()} fc2={model.fc2.weight.numel() if model.fc2 is not None else 0} "
        f"total={sum(p.numel() for p in model.parameters())}"
    )
    print(f"samples: train={n_train} test={n_test}")

    loaded_checkpoint = None
    if args.load_checkpoint is not None:
        loaded_checkpoint = torch.load(args.load_checkpoint, map_location=device)
        checkpoint_arch = loaded_checkpoint.get("arch", args.arch) if isinstance(loaded_checkpoint, dict) else args.arch
        allowed_v6_from_v5 = args.arch == "v6" and checkpoint_arch == "v5"
        if isinstance(loaded_checkpoint, dict) and checkpoint_arch != args.arch and not allowed_v6_from_v5:
            raise ValueError(
                f"checkpoint arch {checkpoint_arch} does not match --arch {args.arch}"
            )
        state = loaded_checkpoint["model_state"] if isinstance(loaded_checkpoint, dict) and "model_state" in loaded_checkpoint else loaded_checkpoint
        model.load_state_dict(state, strict=False)
        if isinstance(loaded_checkpoint, dict):
            set_activation_shifts(
                model,
                int(loaded_checkpoint.get("conv2_shift", args.conv2_shift)),
                int(loaded_checkpoint.get("v4_pool1_shift", args.v4_pool1_shift)),
                int(loaded_checkpoint.get("v4_pool2_shift", args.v4_pool2_shift)),
                int(loaded_checkpoint.get("v4_gap_shift", args.v4_gap_shift)),
            )
        print(f"loaded checkpoint: {args.load_checkpoint}")

    if args.float_only:
        best_float = train(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.epochs,
            base_lr=args.lr,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            warmup_epochs=min(args.warmup_epochs, args.epochs),
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            weight_mode="float",
            activation_mode="float",
            phase_name="float",
            opt_name=opt_name,
            select_weight_mode="float",
            select_activation_mode="float",
        )
        final_float = evaluate_modes(model, test_loader, device, "float", "float")
        final_quant_raw = evaluate_modes(model, test_loader, device, "quantized", "integer")
        bn_folded_for_export = fold_batch_norm_for_export(model)
        if bn_folded_for_export:
            print("folded BatchNorm into conv weights/biases before diagnostic quant eval")
        weight_scales = calibrate_weight_scales(model, args.w_max)
        bias_scales = calibrate_conv_bias_scales(model)
        final_quant = evaluate_modes(model, test_loader, device, "quantized", "integer")
        print("\n=== float-only summary ===")
        print(f"best_float_acc: {best_float}")
        print(f"final_float_acc: {final_float}")
        print(f"diagnostic_q_acc_raw: {final_quant_raw}")
        print(f"diagnostic_q_acc: {final_quant}")
        print(
            "weight_scales: "
            + " ".join(f"{k}={v:.6g}" for k, v in weight_scales.items())
        )
        if bias_scales:
            print(
                "folded_bias_scales: "
                + " ".join(f"{k}={v:.6g}" for k, v in bias_scales.items())
            )
        if args.save_checkpoint is not None:
            final_quant_raw = evaluate_modes(model, test_loader, device, "quantized", "integer")
            args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "arch": args.arch,
                    "model_state": model.state_dict(),
                    "best_float_acc": best_float,
                    "diagnostic_q_acc": final_quant,
                    "w_max": args.w_max,
                    "conv2_shift": model.conv2_shift,
                    "v4_pool1_shift": model.v4_pool1_shift,
                    "v4_pool2_shift": model.v4_pool2_shift,
                    "v4_gap_shift": model.v4_gap_shift,
                    "bn_folded_for_export": bn_folded_for_export,
                    "bias_scales": bias_scales,
                },
                args.save_checkpoint,
            )
            print(f"saved checkpoint: {args.save_checkpoint}")
        return

    if loaded_checkpoint is not None:
        best_float_acc = float(loaded_checkpoint.get("best_float_acc", 0.0)) if isinstance(loaded_checkpoint, dict) else 0.0
        if best_float_acc <= 0.0:
            best_float_acc = evaluate_modes(model, test_loader, device, "float", "float")
    else:
        best_float_acc = train(
            model,
            train_loader,
            test_loader,
            device,
            epochs=relaxed_epochs,
            base_lr=args.lr,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            warmup_epochs=min(args.warmup_epochs, relaxed_epochs),
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            weight_mode="float",
            activation_mode="float",
            phase_name="relaxed",
            opt_name=opt_name,
            select_weight_mode="float",
            select_activation_mode="float",
            report_float_acc=True,
        )
    bn_folded_for_export = fold_batch_norm_for_export(model)
    if bn_folded_for_export:
        print("folded BatchNorm into conv weights/biases before calibration/export")
    precalibration_relaxed_q_acc = evaluate_modes(model, test_loader, device, "quantized", "integer")
    weight_scales = calibrate_weight_scales(model, args.w_max)
    bias_scales = calibrate_conv_bias_scales(model)
    calibrated_q_acc = evaluate_modes(model, test_loader, device, "quantized", "integer")
    shift_sweep_best = None
    shift_sweep_top = []
    qat_best_q_acc = None
    if args.shift_sweep:
        print("\n=== VGG activation shift sweep ===")
        shift_sweep_best, shift_sweep_top = sweep_activation_shifts(
            model,
            test_loader,
            device,
            parse_int_list(args.sweep_conv2),
            parse_int_list(args.sweep_pool1),
            parse_int_list(args.sweep_pool2),
            parse_int_list(args.sweep_gap),
            max_samples=args.sweep_samples,
        )
        for i, row in enumerate(shift_sweep_top, 1):
            print(
                f"  #{i}: q_acc={float(row['q_acc'])*100:.2f}% "
                f"pool1>>>{row['v4_pool1_shift']} pool2>>>{row['v4_pool2_shift']} "
                f"gap>>>{row['v4_gap_shift']} final>>>{row['conv2_shift']}"
            )
        set_activation_shifts(
            model,
            int(shift_sweep_best["conv2_shift"]),
            int(shift_sweep_best["v4_pool1_shift"]),
            int(shift_sweep_best["v4_pool2_shift"]),
            int(shift_sweep_best["v4_gap_shift"]),
        )
        bias_scales = calibrate_conv_bias_scales(model)
        calibrated_q_acc = evaluate_modes(model, test_loader, device, "quantized", "integer")
        shift_sweep_best["full_q_acc"] = calibrated_q_acc
        print(f"  selected full-test q_acc={calibrated_q_acc*100:.2f}%")
    calibrated_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print("\n=== post-training weight calibration ===")
    print(f"best_float_acc={best_float_acc*100:.2f}%")
    print(f"precalibration_relaxed_q_acc={precalibration_relaxed_q_acc*100:.2f}%")
    print("weight_scales: " + " ".join(f"{k}={v:.6g}" for k, v in weight_scales.items()))
    print(f"calibrated_q_acc={calibrated_q_acc*100:.2f}%")
    if args.qat:
        print("\n=== VGG QAT fixed-point fine-tune ===")
        # Keep QAT on the same scaled integer weights that will be written to .mem.
        weight_scales = calibrate_weight_scales(model, args.w_max)
        bias_scales = calibrate_conv_bias_scales(model)
        print(
            "QAT activation scales are hardware-representable integer right shifts: "
            f"pool1={format_pow2_scale(model.v4_pool1_shift)} "
            f"pool2={format_pow2_scale(model.v4_pool2_shift)} "
            f"gap={format_pow2_scale(model.v4_gap_shift)} "
            f"final={format_pow2_scale(model.conv2_shift)}"
        )
        qat_best_q_acc = train(
            model,
            train_loader,
            test_loader,
            device,
            epochs=args.qat_epochs,
            base_lr=args.lr * args.qat_lr_scale,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            warmup_epochs=min(max(1, args.warmup_epochs // 2), args.qat_epochs),
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            weight_mode="quantized",
            activation_mode="integer",
            phase_name="qat",
            opt_name=opt_name,
            select_weight_mode="quantized",
            select_activation_mode="integer",
            report_float_acc=True,
        )
        if qat_best_q_acc > calibrated_q_acc:
            calibrated_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            calibrated_q_acc = qat_best_q_acc
            print(f"QAT improved fixed/export q_acc to {calibrated_q_acc*100:.2f}%")
        else:
            model.load_state_dict(calibrated_state)
            model.set_modes("quantized", "integer")
            print(
                f"QAT best q_acc={qat_best_q_acc*100:.2f}% did not beat "
                f"calibrated q_acc={calibrated_q_acc*100:.2f}%; restored calibrated state"
            )
    best_quant: float | None = None
    if quant_epochs > 0 and loaded_checkpoint is None:
        best_quant = train(
            model,
            train_loader,
            test_loader,
            device,
            epochs=quant_epochs,
            base_lr=args.lr * 0.2,
            temperature=args.temperature,
            label_smoothing=args.label_smoothing,
            warmup_epochs=min(max(1, args.warmup_epochs // 2), quant_epochs),
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            weight_mode="quantized",
            activation_mode="integer",
            phase_name="quant",
            opt_name=opt_name,
        )
    if best_quant is None or calibrated_q_acc >= best_quant:
        model.load_state_dict(calibrated_state)
        model.set_modes("quantized", "integer")
    best = calibrated_q_acc if best_quant is None else max(calibrated_q_acc, best_quant)

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": args.arch,
                "model_state": model.state_dict(),
                "best_float_acc": best_float_acc,
                "best_q_acc": best,
                "calibrated_q_acc": calibrated_q_acc,
                "qat_best_q_acc": qat_best_q_acc,
                "w_max": args.w_max,
                "conv2_shift": model.conv2_shift,
                "v4_pool1_shift": model.v4_pool1_shift,
                "v4_pool2_shift": model.v4_pool2_shift,
                "v4_gap_shift": model.v4_gap_shift,
                "weight_scales": weight_scales,
                "bias_scales": bias_scales,
                "bn_folded_for_export": bn_folded_for_export,
            },
            args.save_checkpoint,
        )
        print(f"saved checkpoint: {args.save_checkpoint}")

    wc1, wc2, wc3, wc4, wc5, wc6, w1, w2 = quantized_weights_int(model, args.w_max)
    if args.arch == "v2":
        arch_note = (
            "v2 keeps 3x26x26 inputs for minimal export/image-layout disruption; "
            "capacity is increased with ReLU conv blocks, wider channels, and Pool2."
        )
    elif args.arch == "v3":
        arch_note = (
            "v3 keeps the v2 3x26x26 image/export contract and adds a third "
            "32-channel 3x3 ReLU convolution after Conv2; RTL scheduling is not yet added."
        )
    elif args.arch == "v4":
        arch_note = (
            "v4 option A implements the SAGE-mappable VGG-lite plan: full 3x32x32 "
            "input, five padding-preserving 3x3/ReLU convolutions, AvgPool2 after "
            "Conv2 and Conv4, GlobalAvgPool8x8, and one FC 64->10 classifier; no BN "
            "or residual adds."
        )
    elif args.arch in {"v5", "v6"}:
        v6_tail = (
            " v6 keeps this exact topology but uses per-output-channel power-of-two "
            "conv weight scaling in PTQ export and writes conv*_w_scale_shift.mem tables."
            if args.arch == "v6"
            else ""
        )
        arch_note = (
            f"{args.arch} option A implements the documented SAGE-mappable VGG-BN plan: full "
            "3x32x32 input, six padding-preserving 3x3/BN/ReLU convolutions with "
            "64/64/128/128/128/128 channels, AvgPool2 after Conv2 and Conv4, "
            "GlobalAvgPool8x8, and one FC 128->10 classifier. BatchNorm is folded "
            "into conv weights/biases before fixed-point calibration, QAT, and export."
            + v6_tail
        )
    else:
        arch_note = "v1 preserves the original MiniCNN-v2-like CIFAR flow."
    info = {
        "version": arch.version,
        "arch": args.arch,
        "arch_note": arch_note,
        "topology": arch.description,
        "dataset": "synthetic" if args.synthetic else "cifar10",
        "best_test_acc": best,
        "best_float_acc": best_float_acc,
        "precalibration_relaxed_q_acc": precalibration_relaxed_q_acc,
        "calibrated_q_acc": calibrated_q_acc,
        "best_quant_stage_q_acc": "not_run" if best_quant is None else best_quant,
        "best_q_acc": best,
        "training_mode": "quantized-only" if args.no_relaxed else "relaxed-then-quantized",
        "init_mode": init_mode,
        "optimizer": opt_name,
        "float_input_scale": args.input_scale,
        "batch_norm_training": arch.batch_norm,
        "bn_folded_for_export": bn_folded_for_export,
        "w_max": args.w_max,
        "conv2_shift": model.conv2_shift,
        "v4_pool1_shift": model.v4_pool1_shift,
        "v4_pool2_shift": model.v4_pool2_shift,
        "v4_gap_shift": model.v4_gap_shift,
        "v4_pool1_scale": format_pow2_scale(model.v4_pool1_shift),
        "v4_pool2_scale": format_pow2_scale(model.v4_pool2_shift),
        "v4_gap_scale": format_pow2_scale(model.v4_gap_shift),
        "final_feature_scale": format_pow2_scale(model.conv2_shift),
        "v4_shift_sweep": args.shift_sweep,
        "v4_shift_sweep_best": "not_run" if shift_sweep_best is None else shift_sweep_best,
        "v4_shift_sweep_top": "not_run" if not shift_sweep_top else shift_sweep_top,
        "qat_enabled": args.qat,
        "qat_epochs": args.qat_epochs if args.qat else 0,
        "qat_lr_scale": args.qat_lr_scale if args.qat else "not_run",
        "qat_best_q_acc": "not_run" if qat_best_q_acc is None else qat_best_q_acc,
        "act_scale_mode": args.act_scale_mode if args.qat else "not_run",
        "act_scale_export_note": (
            "VGG-style QAT/export uses only power-of-two activation scales encoded as integer "
            "right shifts; no arbitrary learned scale or non-hardware operation is exported."
        ),
        "folded_bias_export_note": (
            "v5/v6 folded BN biases are quantized in conv accumulator units using cumulative input "
            "activation scale times current weight scale and exported as signed 32-bit conv*_b.mem. "
            "For v6, current conv weight scale is per output channel."
            if arch.batch_norm
            else "not_used"
        ),
        "v6_ptq_note": (
            "v6 keeps v5 topology unchanged and applies A1 from the v6 guidebook: "
            "per-output-channel power-of-two conv weight scales. Scale exponents are "
            "stored in conv*_w_scale_shift.mem where physical weight scale is 2^shift."
            if args.arch == "v6"
            else "not_used"
        ),
        "activation_shift_note": (
            "v1/v2/v3 use the existing final shift only; VGG-style CIFAR architectures "
            "can optionally add post-pool and post-global-pool shifts selected by --shift-sweep or QAT"
        ),
        **weight_scales,
        **bias_scales,
        "epochs": total_epochs,
        "relaxed_epochs": relaxed_epochs,
        "quant_epochs": quant_epochs,
        "augmentation": args.augment,
        "input_shape": f"3x{arch.input_size}x{arch.input_size}",
        "image_layout": "CHW",
        "pixel_centering": "floor(u8/4)-32",
        "conv_relu": arch.conv_relu,
        "pool_shape": f"AvgPool{arch.pool_kernel} stride {arch.pool_stride}",
        "conv_padding": arch.conv_padding,
        "global_avg_pool": arch.global_avg_pool,
        "single_fc_classifier": arch.single_fc,
        "conv1_shape": f"{arch.conv1_out}x3x3x3",
        "conv1_output_shape": f"{arch.conv1_out}x{arch.conv1_spatial}x{arch.conv1_spatial}",
        "conv2_shape": f"{arch.conv2_out}x{arch.conv1_out}x3x3",
        "conv2_output_shape": f"{arch.conv2_out}x{arch.conv2_spatial}x{arch.conv2_spatial}",
        "conv3_shape": (
            "none"
            if arch.conv3_out is None
            else f"{arch.conv3_out}x{arch.conv2_out}x3x3"
        ),
        "conv3_output_shape": "none" if arch.conv3_out is None else f"{arch.conv3_out}x{arch.conv3_spatial}x{arch.conv3_spatial}",
        "conv4_shape": (
            "none"
            if arch.conv4_out is None
            else f"{arch.conv4_out}x{arch.conv3_out}x3x3"
        ),
        "conv4_output_shape": "none" if arch.conv4_out is None else f"{arch.conv4_out}x{arch.conv4_spatial}x{arch.conv4_spatial}",
        "conv5_shape": (
            "none"
            if arch.conv5_out is None
            else f"{arch.conv5_out}x{arch.conv4_out}x3x3"
        ),
        "conv5_output_shape": "none" if arch.conv5_out is None else f"{arch.conv5_out}x{arch.conv5_spatial}x{arch.conv5_spatial}",
        "conv6_shape": (
            "none"
            if arch.conv6_out is None
            else f"{arch.conv6_out}x{arch.conv5_out}x3x3"
        ),
        "conv6_output_shape": "none" if arch.conv6_out is None else f"{arch.conv6_out}x{arch.conv6_spatial}x{arch.conv6_spatial}",
        "final_conv_output_shape": (
            f"{arch.final_conv_out}x{arch.final_conv_spatial}x{arch.final_conv_spatial}"
        ),
        "global_avg_pool_output_shape": "none" if not arch.global_avg_pool else f"{arch.final_conv_out}",
        "fc1_shape": f"{model.fc1.out_features}x{arch.fc_in}",
        "fc2_shape": (
            "none"
            if model.fc2 is None
            else f"12x{arch.fc_hidden} (rows 10..11 padded zero)"
        ),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "conv1_max": int(np.abs(wc1).max()),
        "conv2_max": int(np.abs(wc2).max()),
        "conv3_max": "not_used" if wc3 is None else int(np.abs(wc3).max()),
        "conv4_max": "not_used" if wc4 is None else int(np.abs(wc4).max()),
        "conv5_max": "not_used" if wc5 is None else int(np.abs(wc5).max()),
        "conv6_max": "not_used" if wc6 is None else int(np.abs(wc6).max()),
        "w1_max": int(np.abs(w1).max()),
        "w2_max": "not_used" if w2 is None else int(np.abs(w2).max()),
        "conv1_nz": int((wc1 != 0).sum()),
        "conv2_nz": int((wc2 != 0).sum()),
        "conv3_nz": "not_used" if wc3 is None else int((wc3 != 0).sum()),
        "conv4_nz": "not_used" if wc4 is None else int((wc4 != 0).sum()),
        "conv5_nz": "not_used" if wc5 is None else int((wc5 != 0).sum()),
        "conv6_nz": "not_used" if wc6 is None else int((wc6 != 0).sum()),
        "w1_nz": int((w1 != 0).sum()),
        "w2_nz": "not_used" if w2 is None else int((w2 != 0).sum()),
        "train_samples": n_train,
        "test_samples": n_test,
    }

    export_weights(wc1, wc2, wc3, wc4, wc5, wc6, w1, w2, model, data_dir)
    export_test_set(model, test_loader, args.export, data_dir, info, device, arch)

    info_txt = "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n"
    (data_dir / "info.txt").write_text(info_txt)
    print("\n=== summary ===")
    print(info_txt)
    print(f"wrote artefacts under {data_dir}/")


if __name__ == "__main__":
    main()
