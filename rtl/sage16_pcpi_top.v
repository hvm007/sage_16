// =============================================================================
// sage16_pcpi_top.v - multi-kernel PicoRV32 PCPI wrapper around sage16_top
//
// Extends the old sage16_pcpi.v (matmul-only, 4 insns) with a MODE register so
// a single set of 4 custom instructions can drive all three kernels.
//
// Custom instructions (R-type, opcode=0b0001011 "custom-0", funct7=0):
//
//   | mnemonic      | funct3 | rs1       | rs2      | rd  | effect                           |
//   |---------------|--------|-----------|----------|-----|----------------------------------|
//   | sage16.wrOp     | 000    | idx (6b)  | val(16b) |  -  | opbuf[idx] <= val                |
//   | sage16.rdC      | 011    | idx (4b)  |  -       | rd  | rd <- c_buf[idx] (zero-ext ACC_W)|
//   | sage16.go       | 010    |  -        |  -       |  -  | start, blocks until done         |
//   | sage16.setMode  | 100    | mode (2b) |  -       |  -  | mode_reg <= mode                 |
//
// opbuf layout (identical to sage16_mmio_top):
//   MODE=0 matmul : [0..15]=A, [16..31]=B
//   MODE=1 conv   : [0..35]=IMG, [36..44]=K
//   MODE=2 quat   : [0..3]=Q1 (w,x,y,z), [4..19]=Q2 flat
//
// Inline-asm helpers (GCC):
//   #define sage16_SETMODE(m)   asm volatile (".insn r 0x0B,4,0,x0,%0,x0"::"r"(m))
//   #define sage16_WROP(i,v)    asm volatile (".insn r 0x0B,0,0,x0,%0,%1"::"r"(i),"r"(v))
//   #define sage16_GO()         asm volatile (".insn r 0x0B,2,0,x0,x0,x0")
//   static inline int sage16_RDC(int i){int r;
//       asm volatile(".insn r 0x0B,3,0,%0,%1,x0":"=r"(r):"r"(i));return r;}
//
// The go-wait FSM uses a rising-edge detector on sage16_done so repeated
// sage16.go calls don't see stale done=1 from a previous kernel (sage16_top
// holds done high across the idle gap between kernels).
// =============================================================================
module sage16_pcpi_top #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter PIPELINE = 1
)(
    input                 clk, rst_n,

    // --- PCPI slave port (connects to PicoRV32) ---
    input                 pcpi_valid,
    input      [31:0]     pcpi_insn,
    input      [31:0]     pcpi_rs1,
    input      [31:0]     pcpi_rs2,
    output reg            pcpi_wr,
    output reg [31:0]     pcpi_rd,
    output reg            pcpi_wait,
    output reg            pcpi_ready
);
    // ----- instruction decode -----
    localparam [6:0] sage16_OPCODE = 7'b0001011;  // RISC-V custom-0

    localparam [2:0] F_WROP = 3'd0,
                     F_GO   = 3'd2,
                     F_RDC  = 3'd3,
                     F_MODE = 3'd4;

    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] funct3 = pcpi_insn[14:12];
    wire [6:0] funct7 = pcpi_insn[31:25];
    wire       is_ours = pcpi_valid &&
                         (opcode == sage16_OPCODE) &&
                         (funct7 == 7'd0);

    wire [5:0]        op_idx = pcpi_rs1[5:0];
    wire [3:0]        c_idx  = pcpi_rs1[3:0];
    wire [1:0]        mode_v = pcpi_rs1[1:0];
    wire [DATA_W-1:0] wr_val = pcpi_rs2[DATA_W-1:0];

    // ----- operand buffer -----
    reg [DATA_W-1:0] opbuf [0:63];

    // ----- mode register -----
    reg [1:0] mode_reg;

    // ----- operand repacking (same layout as sage16_mmio_top) -----
    wire [16*DATA_W-1:0] mm_a_flat;
    wire [16*DATA_W-1:0] mm_b_flat;
    wire [36*DATA_W-1:0] cv_img_flat;
    wire [ 9*DATA_W-1:0] cv_k_flat;
    wire [16*DATA_W-1:0] qt_q2_flat;

    genvar gi;
    generate for(gi=0; gi<16; gi=gi+1) begin : gmm
        assign mm_a_flat  [gi*DATA_W +: DATA_W] = opbuf[gi];
        assign mm_b_flat  [gi*DATA_W +: DATA_W] = opbuf[gi + 16];
        assign qt_q2_flat [gi*DATA_W +: DATA_W] = opbuf[gi + 4];
    end endgenerate
    generate for(gi=0; gi<36; gi=gi+1) begin : gimg
        assign cv_img_flat[gi*DATA_W +: DATA_W] = opbuf[gi];
    end endgenerate
    generate for(gi=0; gi<9; gi=gi+1) begin : gker
        assign cv_k_flat  [gi*DATA_W +: DATA_W] = opbuf[gi + 36];
    end endgenerate

    wire [DATA_W-1:0] qt_q1_w = opbuf[0];
    wire [DATA_W-1:0] qt_q1_x = opbuf[1];
    wire [DATA_W-1:0] qt_q1_y = opbuf[2];
    wire [DATA_W-1:0] qt_q1_z = opbuf[3];

    // ----- sage16_top -----
    reg                 sage16_start;
    wire                sage16_done;
    wire [1:0]          mode_out;
    wire [16*ACC_W-1:0] c_flat;

    sage16_top #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .PIPELINE(PIPELINE)
    ) u_top (
        .clk(clk), .rst_n(rst_n),
        .mode    (mode_reg),
        .start   (sage16_start),
        .done    (sage16_done),
        .mode_out(mode_out),
        .mm_a    (mm_a_flat),
        .mm_b    (mm_b_flat),
        .cv_img  (cv_img_flat),
        .cv_k    (cv_k_flat),
        .qt_q1_w (qt_q1_w),
        .qt_q1_x (qt_q1_x),
        .qt_q1_y (qt_q1_y),
        .qt_q1_z (qt_q1_z),
        .qt_q2   (qt_q2_flat),
        .c_out   (c_flat)
    );

    // ----- rising-edge detector on sage16_done -----
    reg  sage16_done_d;
    wire sage16_done_rise = sage16_done & ~sage16_done_d;

    // ----- sage16.go FSM -----
    localparam [1:0] GO_IDLE = 2'd0,
                     GO_FIRE = 2'd1,
                     GO_WAIT = 2'd2;
    reg [1:0] go_state;

    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            pcpi_wr     <= 0;
            pcpi_rd     <= 32'd0;
            pcpi_wait   <= 0;
            pcpi_ready  <= 0;
            sage16_start  <= 0;
            sage16_done_d <= 0;
            mode_reg    <= 2'd0;
            go_state    <= GO_IDLE;
            for(ii=0; ii<64; ii=ii+1) opbuf[ii] <= 0;
        end else begin
            // default pulse-low
            pcpi_wr     <= 0;
            pcpi_ready  <= 0;
            pcpi_wait   <= 0;
            sage16_start  <= 0;
            sage16_done_d <= sage16_done;

            if(is_ours) begin
                case(funct3)
                // ----- sage16.wrOp : opbuf[idx] <= val, 1 cycle -----
                F_WROP: begin
                    opbuf[op_idx] <= wr_val;
                    pcpi_ready    <= 1;
                end
                // ----- sage16.setMode : mode_reg <= mode_v, 1 cycle -----
                F_MODE: begin
                    mode_reg   <= mode_v;
                    pcpi_ready <= 1;
                end
                // ----- sage16.rdC : rd <= resbuf[idx] (zero-ext), 1 cycle -----
                F_RDC: begin
                    pcpi_rd    <= c_flat[c_idx*ACC_W +: ACC_W];
                    pcpi_wr    <= 1;
                    pcpi_ready <= 1;
                end
                // ----- sage16.go : pulse start, wait until sage16_done rising -----
                F_GO: begin
                    case(go_state)
                    GO_IDLE: begin
                        sage16_start <= 1;
                        pcpi_wait  <= 1;
                        go_state   <= GO_FIRE;
                    end
                    GO_FIRE: begin
                        pcpi_wait <= 1;
                        go_state  <= GO_WAIT;
                    end
                    GO_WAIT: begin
                        if(sage16_done_rise) begin
                            pcpi_wait  <= 0;
                            pcpi_ready <= 1;
                            go_state   <= GO_IDLE;
                        end else begin
                            pcpi_wait <= 1;
                        end
                    end
                    default: go_state <= GO_IDLE;
                    endcase
                end
                default: ;
                endcase
            end else begin
                if(!pcpi_valid) go_state <= GO_IDLE;
            end
        end
    end
endmodule
