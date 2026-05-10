// =============================================================================
// sage16_sys.v
//
// System top: PicoRV32 core + sage16 accelerator tightly coupled via PCPI.
//
// This file WILL NOT COMPILE until you drop picorv32.v into this folder:
//     curl -L -o picorv32.v https://raw.githubusercontent.com/YosysHQ/picorv32/main/picorv32.v
//
// Then build with:
//     iverilog -g2012 -o sys.vvp sage16_sys.v tb_sys.v
//
// The accelerator is exposed as 4 custom R-type instructions under opcode
// 0x0B (custom-0). See sage16_pcpi.v for the instruction encoding and inline-asm
// helper macros.
//
// Memory interface is left as a simple stub (mem_valid/mem_addr/mem_wdata/
// mem_rdata/mem_ready). For FPGA bring-up, connect it to a BRAM initialized
// with your program (hex).
// =============================================================================
`include "pe.v"
`include "sage16_4x4_mac.v"
`include "matmul_sage16.v"
`include "sage16_pcpi.v"
// `include "picorv32.v"   // <-- uncomment after dropping the file in

module sage16_sys #(parameter MEM_WORDS = 4096)(
    input         clk, rst_n,

    // Simple memory bus (to BRAM / testbench model)
    output        mem_valid,
    input         mem_ready,
    output [31:0] mem_addr,
    output [31:0] mem_wdata,
    output [ 3:0] mem_wstrb,
    input  [31:0] mem_rdata,

    // Debug
    output        trap
);
    // -------- PCPI fabric --------
    wire        pcpi_valid;
    wire [31:0] pcpi_insn;
    wire [31:0] pcpi_rs1, pcpi_rs2;
    wire        pcpi_wr;
    wire [31:0] pcpi_rd;
    wire        pcpi_wait, pcpi_ready;

    // -------- PicoRV32 --------
    // Keeping this in a `ifdef so the file compiles without the core.
`ifdef HAVE_PICORV32
    picorv32 #(
        .ENABLE_PCPI(1),
        .ENABLE_MUL(0),
        .ENABLE_DIV(0),
        .BARREL_SHIFTER(0),
        .COMPRESSED_ISA(0),
        .PROGADDR_RESET(32'h0000_0000)
    ) u_core (
        .clk(clk), .resetn(rst_n),
        .trap(trap),
        .mem_valid(mem_valid), .mem_instr(),
        .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata),
        .mem_wstrb(mem_wstrb), .mem_rdata(mem_rdata),

        // PCPI
        .pcpi_valid(pcpi_valid),
        .pcpi_insn (pcpi_insn),
        .pcpi_rs1  (pcpi_rs1),
        .pcpi_rs2  (pcpi_rs2),
        .pcpi_wr   (pcpi_wr),
        .pcpi_rd   (pcpi_rd),
        .pcpi_wait (pcpi_wait),
        .pcpi_ready(pcpi_ready),

        // other optional ports (irq, etc.) tied off
        .irq(32'd0)
    );
`else
    assign mem_valid = 0;
    assign mem_addr  = 0;
    assign mem_wdata = 0;
    assign mem_wstrb = 0;
    assign trap      = 0;
    assign pcpi_valid = 0;
    assign pcpi_insn  = 0;
    assign pcpi_rs1   = 0;
    assign pcpi_rs2   = 0;
`endif

    // -------- sage16 coprocessor --------
    sage16_pcpi #(.DATA_W(16)) u_cop(
        .clk(clk), .rst_n(rst_n),
        .pcpi_valid(pcpi_valid),
        .pcpi_insn (pcpi_insn),
        .pcpi_rs1  (pcpi_rs1),
        .pcpi_rs2  (pcpi_rs2),
        .pcpi_wr   (pcpi_wr),
        .pcpi_rd   (pcpi_rd),
        .pcpi_wait (pcpi_wait),
        .pcpi_ready(pcpi_ready)
    );
endmodule
