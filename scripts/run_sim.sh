#!/usr/bin/env bash
# =============================================================================
#  sage16 simulation runner for Linux / macOS (iverilog + vvp)
#  Mirrors scripts/run_sim.bat target-for-target.
#
#    scripts/run_sim.sh                 -> all quick testbenches (no MNIST/CIFAR payloads)
#    scripts/run_sim.sh matmul conv     -> just those targets
#    scripts/run_sim.sh --list          -> list targets
#
#  Exit code is non-zero if any target fails to compile, crashes, or prints
#  a SUMMARY/result line with fail != 0.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p build

RTL="rtl/pe.v rtl/sage16_4x4_mac.v rtl/matmul_sage16.v"
TOP="rtl/sage16_top.v"

# target -> "extra iverilog flags|source files"
declare -A T=(
  [matmul]="|sim/tb_matmul.v"
  [pcpi]="|rtl/sage16_pcpi.v sim/tb_pcpi.v"
  [mmio]="|rtl/sage16_mmio.v sim/tb_mmio.v"
  [conv]="|rtl/conv3x3_sage16.v sim/tb_conv.v"
  [quat]="|rtl/quat_sage16.v sim/tb_quat.v"
  [quat-vrot]="-DQUAT_VROT_ONLY|rtl/quat_sage16.v sim/tb_quat.v"
  [reconfig]="|$TOP sim/tb_reconfig.v"
  [mmio_top]="|$TOP rtl/sage16_mmio_top.v sim/tb_mmio_top.v"
  [pcpi_top]="|$TOP rtl/sage16_pcpi_top.v sim/tb_pcpi_top.v"
  [tiled]="|$TOP sim/tb_matmul_tiled.v"
  [b2b]="|$TOP sim/tb_reconfig_b2b.v"
  [pairs]="|$TOP sim/tb_reconfig_pairs.v"
  [minicnn]="|$TOP sim/tb_minicnn.v"
  [minicnn-v2]="|$TOP sim/tb_minicnn_v2.v"
  [madgwick]="|$TOP sim/tb_madgwick.v"
  # payload-dependent / long runs (not part of the default set)
  [minicnn-mnist]="-DREAL_MNIST|$TOP sim/tb_minicnn.v"
  [minicnn-v2-mnist]="-DREAL_MNIST|$TOP sim/tb_minicnn_v2.v"
  [cifar10-v5-smoke]="-DREAL_CIFAR10_V5 -Ptb_cifar10_v5.N_IMG=2 -Ptb_cifar10_v5.POOL1_SHIFT=0 -Ptb_cifar10_v5.POOL2_SHIFT=0 -Ptb_cifar10_v5.GAP_SHIFT=0 -Ptb_cifar10_v5.FINAL_SHIFT=4|$TOP sim/tb_cifar10_v5.v"
)
QUICK="matmul pcpi mmio conv quat quat-vrot reconfig mmio_top pcpi_top tiled b2b pairs minicnn minicnn-v2 madgwick"

need_file() { [ -f "$1" ] || { echo "  missing $1 -- $2"; return 1; }; }

if [ "${1:-}" = "--list" ]; then
  echo "quick (default): $QUICK"
  echo "payload/long   : minicnn-mnist minicnn-v2-mnist cifar10-v5-smoke"
  exit 0
fi

command -v iverilog >/dev/null && command -v vvp >/dev/null || {
  echo "iverilog/vvp not found on PATH (Icarus Verilog 11+ required)"; exit 2; }

targets="${*:-$QUICK}"
failed=()
for t in $targets; do
  spec="${T[$t]:-}"
  if [ -z "$spec" ]; then echo "unknown target: $t (try --list)"; exit 2; fi
  flags="${spec%%|*}"; files="${spec#*|}"
  case "$t" in
    minicnn-mnist)    need_file data/images.mem "run: python scripts/train_minicnn.py" || { failed+=("$t"); continue; } ;;
    minicnn-v2-mnist) need_file data/v2/images.mem "run: python scripts/train_minicnn_v2.py" || { failed+=("$t"); continue; } ;;
    cifar10-v5-smoke) need_file data/cifar10_v5/images.mem "see README: CIFAR-10 full flow" || { failed+=("$t"); continue; } ;;
  esac
  echo "=== $t"
  log="build/sim_${t//-/_}.log"
  # shellcheck disable=SC2086
  if ! iverilog -g2012 $flags -o "build/sim_${t//-/_}.vvp" $RTL $files; then
    echo "  COMPILE FAILED"; failed+=("$t"); continue
  fi
  vvp -n "build/sim_${t//-/_}.vvp" | tee "$log" | grep -iE "summary|pass=|fail=|transitions|cycles" | tail -6
  if [ "${PIPESTATUS[0]}" -ne 0 ] || grep -qiE "fail=[1-9]|fail *: *[1-9]|FATAL|ERROR" "$log"; then
    failed+=("$t")
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then echo "ALL PASSED: $targets"; exit 0; fi
echo "FAILED: ${failed[*]}"; exit 1
