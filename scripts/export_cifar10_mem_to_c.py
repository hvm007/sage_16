#!/usr/bin/env python3
"""
export_cifar10_mem_to_c.py
==========================
Load CIFAR .mem artifacts and emit a single C header for firmware-side use.

Usage:
  python scripts/export_cifar10_mem_to_c.py
  python scripts/export_cifar10_mem_to_c.py --n-img 16
  python scripts/export_cifar10_mem_to_c.py --arch v2 --n-img 16
  python scripts/export_cifar10_mem_to_c.py --arch v3 --n-img 16
  python scripts/export_cifar10_mem_to_c.py --arch v4 --n-img 16
  python scripts/export_cifar10_mem_to_c.py --arch v5 --n-img 16
  python scripts/export_cifar10_mem_to_c.py --arch v6 --n-img 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "cifar10_v1"


ARCH_SPECS = {
    "v1": {
        "version": "cifar10_v1",
        "data_dir": ROOT / "data" / "cifar10_v1",
        "input_size": 26,
        "conv1_out": 8,
        "conv2_out": 16,
        "conv3_out": 0,
        "conv4_out": 0,
        "conv5_out": 0,
        "conv6_out": 0,
        "pool": 4,
        "fc_hidden": 32,
        "fc_in": 256,
        "single_fc": False,
    },
    "v2": {
        "version": "cifar10_v2",
        "data_dir": ROOT / "data" / "cifar10_v2",
        "input_size": 26,
        "conv1_out": 16,
        "conv2_out": 32,
        "conv3_out": 0,
        "conv4_out": 0,
        "conv5_out": 0,
        "conv6_out": 0,
        "pool": 2,
        "fc_hidden": 32,
        "fc_in": 3200,
        "single_fc": False,
    },
    "v3": {
        "version": "cifar10_v3",
        "data_dir": ROOT / "data" / "cifar10_v3",
        "input_size": 26,
        "conv1_out": 16,
        "conv2_out": 32,
        "conv3_out": 32,
        "conv4_out": 0,
        "conv5_out": 0,
        "conv6_out": 0,
        "pool": 2,
        "fc_hidden": 32,
        "fc_in": 2048,
        "single_fc": False,
    },
    "v4": {
        "version": "cifar10_v4",
        "data_dir": ROOT / "data" / "cifar10_v4",
        "input_size": 32,
        "conv1_out": 32,
        "conv2_out": 32,
        "conv3_out": 64,
        "conv4_out": 64,
        "conv5_out": 64,
        "conv6_out": 0,
        "pool": 2,
        "fc_hidden": 10,
        "fc_in": 64,
        "single_fc": True,
    },
    "v5": {
        "version": "cifar10_v5",
        "data_dir": ROOT / "data" / "cifar10_v5",
        "input_size": 32,
        "conv1_out": 64,
        "conv2_out": 64,
        "conv3_out": 128,
        "conv4_out": 128,
        "conv5_out": 128,
        "conv6_out": 128,
        "pool": 2,
        "fc_hidden": 10,
        "fc_in": 128,
        "single_fc": True,
        "folded_bn_bias": True,
    },
    "v6": {
        "version": "cifar10_v6",
        "data_dir": ROOT / "data" / "cifar10_v6",
        "input_size": 32,
        "conv1_out": 64,
        "conv2_out": 64,
        "conv3_out": 128,
        "conv4_out": 128,
        "conv5_out": 128,
        "conv6_out": 128,
        "pool": 2,
        "fc_hidden": 10,
        "fc_in": 128,
        "single_fc": True,
        "folded_bn_bias": True,
        "per_channel_conv_scales": True,
    },
}


def hex16_to_i16(s: str) -> int:
    s = s.strip()
    if not s:
        raise ValueError("empty line in .mem")
    v = int(s, 16) & 0xFFFF
    if v & 0x8000:
        v -= 0x10000
    return v


def read_mem_int16(path: Path) -> list[int]:
    return [hex16_to_i16(line) for line in path.read_text().splitlines() if line.strip()]


def hex32_to_i32(s: str) -> int:
    s = s.strip()
    if not s:
        raise ValueError("empty line in .mem")
    v = int(s, 16) & 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v


def read_mem_int32(path: Path) -> list[int]:
    return [hex32_to_i32(line) for line in path.read_text().splitlines() if line.strip()]


def read_optional_mem_int16(path: Path, expected_len: int) -> list[int]:
    if not path.exists():
        return [0] * expected_len
    values = read_mem_int16(path)
    if len(values) != expected_len:
        raise SystemExit(f"{path.name} size mismatch: {len(values)}")
    return values


def read_optional_mem_int32(path: Path, expected_len: int) -> list[int]:
    if not path.exists():
        return [0] * expected_len
    values = read_mem_int32(path)
    if len(values) != expected_len:
        raise SystemExit(f"{path.name} size mismatch: {len(values)}")
    return values


def read_info(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    info = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def read_optional_decimal_lines(path: Path, expected_max: int | None = None) -> list[int]:
    if not path.exists():
        return []
    values = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if expected_max is not None and len(values) < expected_max:
        raise SystemExit(f"{path.name} shorter than requested image count")
    return values


def emit_array(f, ctype: str, name: str, values: list[int], per_line: int = 8) -> None:
    f.write(f"static const {ctype} {name}[{len(values)}] = {{\n")
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        f.write("  " + ", ".join(str(v) for v in chunk) + ",\n")
    f.write("};\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCH_SPECS), default="v1", help="Artifact architecture")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Artifact directory; default is data/cifar10_<arch>",
    )
    ap.add_argument("--n-img", type=int, default=None, help="First N images/labels only")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "firmware" / "minicnn_cifar10_data.h",
    )
    args = ap.parse_args()

    spec = ARCH_SPECS[args.arch]
    d = args.data_dir if args.data_dir is not None else spec["data_dir"]
    info = read_info(d / "info.txt")
    if info and info.get("arch", args.arch) != args.arch:
        raise SystemExit(f"info.txt arch={info.get('arch')} but --arch={args.arch}")

    c1 = read_mem_int16(d / "conv1_k.mem")
    c2 = read_mem_int16(d / "conv2_k.mem")
    c3 = read_mem_int16(d / "conv3_k.mem") if spec["conv3_out"] else []
    c4 = read_mem_int16(d / "conv4_k.mem") if spec["conv4_out"] else []
    c5 = read_mem_int16(d / "conv5_k.mem") if spec["conv5_out"] else []
    c6 = read_mem_int16(d / "conv6_k.mem") if spec.get("conv6_out", 0) else []
    b1 = read_optional_mem_int32(d / "conv1_b.mem", spec["conv1_out"]) if spec.get("folded_bn_bias") else []
    b2 = read_optional_mem_int32(d / "conv2_b.mem", spec["conv2_out"]) if spec.get("folded_bn_bias") else []
    b3 = read_optional_mem_int32(d / "conv3_b.mem", spec["conv3_out"]) if spec.get("folded_bn_bias") and spec["conv3_out"] else []
    b4 = read_optional_mem_int32(d / "conv4_b.mem", spec["conv4_out"]) if spec.get("folded_bn_bias") and spec["conv4_out"] else []
    b5 = read_optional_mem_int32(d / "conv5_b.mem", spec["conv5_out"]) if spec.get("folded_bn_bias") and spec["conv5_out"] else []
    b6 = read_optional_mem_int32(d / "conv6_b.mem", spec["conv6_out"]) if spec.get("folded_bn_bias") and spec.get("conv6_out", 0) else []
    s1 = read_optional_mem_int16(d / "conv1_w_scale_shift.mem", spec["conv1_out"]) if spec.get("per_channel_conv_scales") else []
    s2 = read_optional_mem_int16(d / "conv2_w_scale_shift.mem", spec["conv2_out"]) if spec.get("per_channel_conv_scales") else []
    s3 = read_optional_mem_int16(d / "conv3_w_scale_shift.mem", spec["conv3_out"]) if spec.get("per_channel_conv_scales") and spec["conv3_out"] else []
    s4 = read_optional_mem_int16(d / "conv4_w_scale_shift.mem", spec["conv4_out"]) if spec.get("per_channel_conv_scales") and spec["conv4_out"] else []
    s5 = read_optional_mem_int16(d / "conv5_w_scale_shift.mem", spec["conv5_out"]) if spec.get("per_channel_conv_scales") and spec["conv5_out"] else []
    s6 = read_optional_mem_int16(d / "conv6_w_scale_shift.mem", spec["conv6_out"]) if spec.get("per_channel_conv_scales") and spec.get("conv6_out", 0) else []
    w1 = read_mem_int16(d / "w1.mem")
    w2 = [] if spec["single_fc"] else read_mem_int16(d / "w2.mem")
    img_flat = read_mem_int16(d / "images.mem")
    labels = read_mem_int16(d / "labels.mem")
    preds = read_optional_decimal_lines(d / "preds.txt")

    expected_c1 = spec["conv1_out"] * 3 * 9
    expected_c2 = spec["conv2_out"] * spec["conv1_out"] * 9
    expected_c3 = spec["conv3_out"] * spec["conv2_out"] * 9
    expected_c4 = spec["conv4_out"] * spec["conv3_out"] * 9
    expected_c5 = spec["conv5_out"] * spec["conv4_out"] * 9
    expected_c6 = spec.get("conv6_out", 0) * spec["conv5_out"] * 9
    expected_w1 = spec["fc_hidden"] * spec["fc_in"]
    expected_w2 = 0 if spec["single_fc"] else 12 * spec["fc_hidden"]
    if len(c1) != expected_c1 or len(c2) != expected_c2:
        raise SystemExit(f"conv size mismatch: c1={len(c1)} c2={len(c2)}")
    if len(c3) != expected_c3:
        raise SystemExit(f"conv3 size mismatch: c3={len(c3)}")
    if len(c4) != expected_c4:
        raise SystemExit(f"conv4 size mismatch: c4={len(c4)}")
    if len(c5) != expected_c5:
        raise SystemExit(f"conv5 size mismatch: c5={len(c5)}")
    if len(c6) != expected_c6:
        raise SystemExit(f"conv6 size mismatch: c6={len(c6)}")
    if len(w1) != expected_w1 or len(w2) != expected_w2:
        raise SystemExit(f"FC size mismatch: w1={len(w1)} w2={len(w2)}")

    per_img = 3 * spec["input_size"] * spec["input_size"]
    n_file = len(img_flat) // per_img
    if len(img_flat) != n_file * per_img:
        raise SystemExit(f"images.mem length {len(img_flat)} not multiple of {per_img}")
    if len(labels) < n_file:
        raise SystemExit("labels.mem shorter than image count")

    n_img = args.n_img if args.n_img is not None else n_file
    n_img = min(n_img, n_file, len(labels))
    img_flat = img_flat[: n_img * per_img]
    labels = labels[:n_img]
    preds = preds[:n_img]

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(
            "/* Auto-generated by scripts/export_cifar10_mem_to_c.py - do not edit. */\n"
            "#ifndef MINICNN_CIFAR10_DATA_H\n"
            "#define MINICNN_CIFAR10_DATA_H\n"
            "#include <stdint.h>\n\n"
        )
        f.write(
            f"/* Same values as data/{d.name}/*.mem */\n"
            f"#define MINICNN_CIFAR10_ARCH_{args.arch.upper()} 1\n"
            f"#define MINICNN_CIFAR10_N_IMG ({n_img})\n\n"
            f"#define MINICNN_CIFAR10_INPUT_SIZE ({spec['input_size']})\n"
            f"#define MINICNN_CIFAR10_CONV1_OUT ({spec['conv1_out']})\n"
            f"#define MINICNN_CIFAR10_CONV2_OUT ({spec['conv2_out']})\n"
            f"#define MINICNN_CIFAR10_CONV3_OUT ({spec['conv3_out']})\n"
            f"#define MINICNN_CIFAR10_CONV4_OUT ({spec['conv4_out']})\n"
            f"#define MINICNN_CIFAR10_CONV5_OUT ({spec['conv5_out']})\n"
            f"#define MINICNN_CIFAR10_CONV6_OUT ({spec.get('conv6_out', 0)})\n"
            f"#define MINICNN_CIFAR10_HAS_FOLDED_BN_BIAS ({1 if spec.get('folded_bn_bias') else 0})\n"
            f"#define MINICNN_CIFAR10_HAS_PER_CHANNEL_CONV_SCALES ({1 if spec.get('per_channel_conv_scales') else 0})\n"
            f"#define MINICNN_CIFAR10_POOL ({spec['pool']})\n"
            f"#define MINICNN_CIFAR10_FC_IN ({spec['fc_in']})\n"
            f"#define MINICNN_CIFAR10_FC_HIDDEN ({spec['fc_hidden']})\n\n"
        )
        emit_array(f, "int16_t", "minicnn_cifar10_conv1_k", c1)
        emit_array(f, "int16_t", "minicnn_cifar10_conv2_k", c2)
        if c3:
            emit_array(f, "int16_t", "minicnn_cifar10_conv3_k", c3)
        if c4:
            emit_array(f, "int16_t", "minicnn_cifar10_conv4_k", c4)
        if c5:
            emit_array(f, "int16_t", "minicnn_cifar10_conv5_k", c5)
        if c6:
            emit_array(f, "int16_t", "minicnn_cifar10_conv6_k", c6)
        if spec.get("folded_bn_bias"):
            emit_array(f, "int32_t", "minicnn_cifar10_conv1_b", b1)
            emit_array(f, "int32_t", "minicnn_cifar10_conv2_b", b2)
            if b3:
                emit_array(f, "int32_t", "minicnn_cifar10_conv3_b", b3)
            if b4:
                emit_array(f, "int32_t", "minicnn_cifar10_conv4_b", b4)
            if b5:
                emit_array(f, "int32_t", "minicnn_cifar10_conv5_b", b5)
            if b6:
                emit_array(f, "int32_t", "minicnn_cifar10_conv6_b", b6)
        if spec.get("per_channel_conv_scales"):
            emit_array(f, "int16_t", "minicnn_cifar10_conv1_w_scale_shift", s1)
            emit_array(f, "int16_t", "minicnn_cifar10_conv2_w_scale_shift", s2)
            if s3:
                emit_array(f, "int16_t", "minicnn_cifar10_conv3_w_scale_shift", s3)
            if s4:
                emit_array(f, "int16_t", "minicnn_cifar10_conv4_w_scale_shift", s4)
            if s5:
                emit_array(f, "int16_t", "minicnn_cifar10_conv5_w_scale_shift", s5)
            if s6:
                emit_array(f, "int16_t", "minicnn_cifar10_conv6_w_scale_shift", s6)
        emit_array(f, "int16_t", "minicnn_cifar10_w1", w1)
        if w2:
            emit_array(f, "int16_t", "minicnn_cifar10_w2", w2)
        emit_array(f, "int16_t", "minicnn_cifar10_img", img_flat)
        emit_array(f, "int16_t", "minicnn_cifar10_labels", labels)
        if preds:
            emit_array(f, "int16_t", "minicnn_cifar10_preds", preds)
        f.write("#endif\n")
    print(f"wrote {out} (N_IMG={n_img}, img_words={len(img_flat)})")


if __name__ == "__main__":
    main()
