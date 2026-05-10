# =============================================================================
#  Vivado synthesis script for the sage16 project
#
#  Usage (from project root):
#      vivado -mode batch -source scripts/synth.tcl
#
#  All reports land in the reports/ folder.
# =============================================================================

create_project -in_memory -part xc7a35tcpg236-1

# --- RTL sources, in dependency order ---------------------------------------
read_verilog -sv {rtl/pe.v}
read_verilog -sv {rtl/sage16_4x4_mac.v}
read_verilog -sv {rtl/matmul_sage16.v}
read_verilog -sv {rtl/sage16_pcpi.v}
read_verilog -sv {rtl/sage16_mmio.v}

# =============================================================================
# Pick ONE synth target (comment/uncomment):
#
#   sage16_4x4_mac  : bare array only (smallest, for area/timing baseline)
#   matmul_sage16   : array + FSM driver (flat I/O - won't fit on 106 IOBs)
#   sage16_pcpi     : array + FSM + PCPI (RISC-V custom-instruction front-end)
#   sage16_mmio     : array + FSM + 32-bit register bus (FPGA-DEPLOYABLE)
#
# For real FPGA bring-up use sage16_mmio.
# For the ICCD area/power numbers use sage16_4x4_mac (isolates the accelerator
# from the interface wrapper).
# =============================================================================
synth_design -top sage16_mmio -flatten_hierarchy full

create_clock -period 5.000 -name clk [get_ports clk]

file mkdir reports
report_utilization     -file {reports/util_report.txt}
report_timing          -file {reports/timing_report.txt}
report_timing_summary  -file {reports/timing_summary.txt}

puts "DONE - reports written to reports/"
