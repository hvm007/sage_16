// =============================================================================
// sage16_mmio_top.v - multi-kernel MMIO wrapper around sage16_top
//
// Why this exists
// ---------------
// sage16_mmio wraps matmul_sage16 and only exposes A/B buffers. The new shared
// fabric sage16_top needs a larger interface because the three kernels have
// different operand sets:
//   matmul : A[16], B[16]
//   conv   : IMG[36], K[9]
//   quat   : Q1[4], Q2[16]
//
// We use a single 64-word operand buffer and a 2-bit MODE register. The
// buffer's meaning depends on MODE; software writes the right bytes into
// the right slots, sets MODE, pulses CTRL.start, and polls STATUS.
//
// Register map (byte addresses, bus is word-addressed addr[9:2])
// ----------------------------------------------------------------
//   0x000..0x0FC   OPBUF[0..63]   (W) 32b words, lower 16b used
//                       MODE=0 (matmul)  : [0..15]=A, [16..31]=B
//                       MODE=1 (conv3x3) : [0..35]=IMG, [36..44]=K
//                       MODE=2 (quat)    : [0..3]=Q1 (w,x,y,z),
//                                          [4..19]=Q2 (4 quats*4 comp)
//   0x100..0x13C   RESBUF[0..15]  (R) 32b signed/unsigned per mode
//   0x200          MODE           (W) bits[1:0]
//   0x204          CTRL           (W) bit0 = start pulse
//   0x208          STATUS         (R) bit0=done, bit1=busy, bits[3:2]=mode
//
// Port count: clk(1)+rst_n(1)+addr[9:0](10)+wdata(32)+rdata(32)+we+re+ack+irq
//           = 80 pins -- fits on any dev board.
// =============================================================================
module sage16_mmio_top #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter PIPELINE = 1,
    parameter ADDR_W   = 10
)(
    input                   clk, rst_n,

    // Simple synchronous register bus (drop-in for AXI-Lite / Wishbone bridge).
    input      [ADDR_W-1:0] addr,          // byte addr; we decode addr[ADDR_W-1:2]
    input      [31:0]       wdata,
    output reg [31:0]       rdata,
    input                   we,
    input                   re,
    output reg              ack,

    output                  irq            // 1-cycle pulse when a kernel finishes
);
    // ----- address decode (word index, up to 256 words) -----
    wire [7:0] widx = addr[ADDR_W-1:2];

    wire is_opbuf_w   = we && (widx <  8'd64);                 // 0x000..0x0FC
    wire is_resbuf_r  = re && (widx >= 8'd64) && (widx < 8'd80); // 0x100..0x13C
    wire is_mode_w    = we && (widx == 8'd128);                // 0x200
    wire is_ctrl_w    = we && (widx == 8'd129);                // 0x204
    wire is_status_r  = re && (widx == 8'd130);                // 0x208

    wire [5:0] op_idx  = widx[5:0];
    wire [3:0] res_idx = widx[3:0];

    // ----- operand buffer (64 words; lower DATA_W bits used) -----
    reg [DATA_W-1:0] opbuf [0:63];

    // ----- result snapshot (16 words, full ACC_W) -----
    reg [ACC_W-1:0] resbuf [0:15];

    // ----- mode and control -----
    // done_r is a "completion-since-last-start" latch:
    //   - cleared when we pulse sage16_start
    //   - set on sage16_done rising edge
    // This is essential for multi-kernel sequences, because sage16_top's
    // `done` signal stays high across the idle gap between kernels; if we
    // sampled it directly the host's STATUS poll would see leftover done=1
    // from the previous kernel and skip the wait.
    reg [1:0] mode_reg;
    reg       sage16_start;
    reg       busy;
    reg       done_r;           // sticky "result ready" flag
    reg       sage16_done_d;      // delayed sage16_done for edge detect
    wire      sage16_done_rise = sage16_done & ~sage16_done_d;

    assign irq = sage16_done_rise;

    // ----- operand repacking for sage16_top -----
    // Each mode uses a different subset of opbuf[]. sage16_top reads every
    // operand port every cycle, so we wire all three layouts simultaneously
    // and let the mode-aware FSM inside sage16_top use the right one.
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

    // quaternion: Q1 = (w,x,y,z) in opbuf[0..3]
    wire [DATA_W-1:0] qt_q1_w = opbuf[0];
    wire [DATA_W-1:0] qt_q1_x = opbuf[1];
    wire [DATA_W-1:0] qt_q1_y = opbuf[2];
    wire [DATA_W-1:0] qt_q1_z = opbuf[3];

    // ----- sage16 shared-fabric core -----
    wire [1:0]          mode_out;
    wire                sage16_done;
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

    // ----- MMIO bus behaviour -----
    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            sage16_start  <= 0;
            busy        <= 0;
            done_r      <= 0;
            sage16_done_d <= 0;
            mode_reg    <= 2'd0;
            ack         <= 0;
            rdata       <= 32'd0;
            for(ii=0; ii<64; ii=ii+1) opbuf[ii]  <= 0;
            for(ii=0; ii<16; ii=ii+1) resbuf[ii] <= 0;
        end else begin
            sage16_start  <= 0;
            ack         <= we | re;
            sage16_done_d <= sage16_done;

            // --- writes ---
            if(is_opbuf_w) opbuf[op_idx] <= wdata[DATA_W-1:0];
            if(is_mode_w)  mode_reg     <= wdata[1:0];
            if(is_ctrl_w && wdata[0] && !busy) begin
                sage16_start <= 1;
                busy       <= 1;
                done_r     <= 0;   // invalidate previous kernel's result flag
            end

            // --- snapshot result on sage16_done rising edge ---
            if(sage16_done_rise) begin
                busy   <= 0;
                done_r <= 1;
                for(ii=0; ii<16; ii=ii+1)
                    resbuf[ii] <= c_flat[ii*ACC_W +: ACC_W];
            end

            // --- reads ---
            rdata <= 32'd0;
            if(is_resbuf_r)
                rdata <= resbuf[res_idx];
            else if(is_status_r)
                rdata <= {26'd0, mode_out, 2'd0, busy, done_r};
        end
    end
endmodule
