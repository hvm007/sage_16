# =============================================================================
# synth_top.tcl - parametric multi-top Vivado flow
#
# Synthesizes, places, routes and reports area/timing/power for ONE top module.
# Designed to be called repeatedly (once per top) by scripts/synth_all.bat.
#
# Usage (from project root):
#
#   vivado -mode batch -source scripts/synth_top.tcl -tclargs <TOP> [period] [mode]
#
# Arguments:
#   TOP    : name of the top module to synthesize (required)
#   period : target clock period in ns      (default 5.000 => 200 MHz)
#   mode   : "ooc" or "deploy"              (default "ooc")
#              ooc    -> out-of-context: no IOB constraints, realistic internal
#                        logic numbers, works for modules with wide flat ports
#              deploy -> full flow with IOB placement: only usable for modules
#                        whose port count fits on the target device (sage16_mmio
#                        family)
#
# Example:
#   vivado -mode batch -source scripts/synth_top.tcl -tclargs sage16_top 5.000 ooc
#
# Basys 3 / board bitstream (requires constraints/basys3.xdc with pin LOC + create_clock):
#   vivado -mode batch -source scripts/synth_top.tcl -tclargs sage16_mmio_top 10.000 deploy
#   -> reports/<TOP>/<TOP>.bit
#
# Outputs land in reports/<TOP>/:
#   util.rpt        post-route utilization (paper's area table source)
#   timing.rpt      post-route timing_summary (WNS, Fmax)
#   timing_top5.rpt top 5 critical paths
#   power.rpt       vector-less power estimate
#   util_synth.rpt  post-synth utilization (pre-implementation)
#   timing_synth.rpt post-synth timing
#   post_synth.dcp  checkpoint after synth_design
#   post_route.dcp  checkpoint after route_design
# =============================================================================

if {[llength $argv] < 1} {
    puts "ERROR: synth_top.tcl needs at least one arg (top module name)"
    puts "Usage: -tclargs <TOP> \[period_ns\] \[mode\]"
    exit 1
}

set TOP    [lindex $argv 0]
if {[llength $argv] >= 2} { set PERIOD [lindex $argv 1] } else { set PERIOD 5.000 }
if {[llength $argv] >= 3} { set MODE   [lindex $argv 2] } else { set MODE "ooc" }
set OUTDIR "reports/$TOP"

file mkdir $OUTDIR
puts "-------------------------------------------------------------------"
puts " synth_top.tcl"
puts "   TOP    = $TOP"
puts "   PERIOD = $PERIOD ns  (Fmax target = [format %.1f [expr 1000.0/$PERIOD]] MHz)"
puts "   MODE   = $MODE"
puts "   OUTDIR = $OUTDIR"
puts "-------------------------------------------------------------------"

# Target device: Artix-7 xc7a35t on the Basys-3 / Arty-A7 boards.
create_project -in_memory -part xc7a35tcpg236-1

set USE_BOARD_XDC 0
if {$MODE eq "deploy" && [file exists constraints/basys3.xdc]} {
    set USE_BOARD_XDC 1
}

# --- RTL sources in dependency order -----------------------------------------
read_verilog -sv {rtl/pe.v}
read_verilog -sv {rtl/sage16_4x4_mac.v}
read_verilog -sv {rtl/matmul_sage16.v}
read_verilog -sv {rtl/conv3x3_sage16.v}
read_verilog -sv {rtl/quat_sage16.v}
read_verilog -sv {rtl/sage16_top.v}
read_verilog -sv {rtl/sage16_pcpi.v}
read_verilog -sv {rtl/sage16_mmio.v}
# new wrappers (present after Day 1b/2a)
if {[file exists rtl/sage16_mmio_top.v]} { read_verilog -sv {rtl/sage16_mmio_top.v} }
if {[file exists rtl/sage16_pcpi_top.v]} { read_verilog -sv {rtl/sage16_pcpi_top.v} }

# Board pinout / timing (Basys 3, etc.). Must be before synth in deploy mode.
if {$USE_BOARD_XDC} {
    read_xdc constraints/basys3.xdc
}

# --- synth -------------------------------------------------------------------
if {$MODE eq "ooc"} {
    synth_design -top $TOP -flatten_hierarchy full -mode out_of_context
} else {
    synth_design -top $TOP -flatten_hierarchy full
}

# If basys3.xdc already defines create_clock on clk, skip duplicate constraint.
if {!$USE_BOARD_XDC} {
    create_clock -period $PERIOD -name clk [get_ports clk]
}

write_checkpoint       -force $OUTDIR/post_synth.dcp
report_utilization     -file  $OUTDIR/util_synth.rpt
report_timing_summary  -file  $OUTDIR/timing_synth.rpt

# --- implementation ----------------------------------------------------------
opt_design
place_design
route_design

# --- post-route reports (these are the paper numbers) -----------------------
write_checkpoint       -force $OUTDIR/post_route.dcp
report_utilization     -file  $OUTDIR/util.rpt
report_timing_summary  -file  $OUTDIR/timing.rpt
report_timing          -max_paths 5 -nworst 5 -file $OUTDIR/timing_top5.rpt
report_power           -file  $OUTDIR/power.rpt
report_design_analysis -complexity -timing -file $OUTDIR/design_analysis.rpt

if {$MODE eq "deploy"} {
    write_bitstream -force $OUTDIR/${TOP}.bit
    puts "   wrote $OUTDIR/${TOP}.bit"
}

puts ""
puts "==================================================================="
puts " DONE: $TOP"
puts "   see $OUTDIR/{util,timing,power}.rpt"
if {$MODE eq "deploy"} {
    puts "   bitstream: $OUTDIR/${TOP}.bit"
}
puts "==================================================================="
