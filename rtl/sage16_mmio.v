// =============================================================================
// sage16_mmio.v  — Memory-Mapped I/O wrapper for FPGA deployment
//
// Why this exists
// ---------------
// The flat matmul_sage16 top has 16*16 + 16*16 + 16*32 + 3 = 770 port bits.
// The default Artix-7 xc7a35t only has 106 IOBs — you can't bond out that
// many signals. This wrapper turns the accelerator into a tiny register file
// accessible through a 32-bit bus, so you can actually deploy it on any
// FPGA (and later tie it to an AXI-Lite bridge for a Zynq / PYNQ flow).
//
// Register map (byte addresses; bus is word-addressed, addr[6:2])
//
//   0x00 .. 0x3C   A[0] .. A[15]     (W, 32b  — only lower DATA_W used)
//   0x40 .. 0x7C   B[0] .. B[15]     (W, 32b)
//   0x80 .. 0xBC   C[0] .. C[15]     (R, ACC_W = 32b)
//   0xC0           CTRL              (W, bit0=start pulse)
//   0xC4           STATUS            (R, bit0=done, bit1=busy)
//
// Port count (top IOBs)
// ---------------------
//   clk(1) + rst_n(1) + addr[6:0](7) + wdata[31:0](32)
// + rdata[31:0](32) + we(1) + re(1) + ack(1) + irq(1) = 76 pins.
// Fits comfortably on any dev board.
// =============================================================================
module sage16_mmio #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter PIPELINE = 1
)(
    input               clk, rst_n,

    // Simple synchronous register bus (drop-in for AXI-Lite / Wishbone front-end)
    input       [7:0]   addr,         // byte addr; we decode addr[7:2] (6b word idx)
    input       [31:0]  wdata,
    output reg  [31:0]  rdata,
    input               we,
    input               re,
    output reg          ack,

    output              irq           // pulses high for 1 cycle when matmul done
);
    // ----- address decode (word index) -----
    wire [5:0]  widx = addr[7:2];
    wire        is_A_write   = we && (widx <  6'h10);                 // 0x00..0x3C
    wire        is_B_write   = we && (widx >= 6'h10 && widx < 6'h20); // 0x40..0x7C
    wire        is_C_read    = re && (widx >= 6'h20 && widx < 6'h30); // 0x80..0xBC
    wire        is_CTRL_wr   = we && (widx == 6'h30);                 // 0xC0
    wire        is_STATUS_rd = re && (widx == 6'h31);                 // 0xC4

    wire [3:0]  ab_idx = widx[3:0];
    wire [3:0]  c_idx  = widx[3:0];

    // ----- A/B buffers -----
    reg [DATA_W-1:0] a_buf [0:15];
    reg [DATA_W-1:0] b_buf [0:15];

    wire [16*DATA_W-1:0] a_flat, b_flat;
    wire [16*ACC_W-1:0]  c_flat;
    genvar g;
    generate for(g=0; g<16; g=g+1) begin : gbuf
        assign a_flat[g*DATA_W +: DATA_W] = a_buf[g];
        assign b_flat[g*DATA_W +: DATA_W] = b_buf[g];
    end endgenerate

    // ----- accelerator core -----
    reg  sage16_start;
    wire sage16_done;

    matmul_sage16 #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .PIPELINE(PIPELINE)
    ) u_mm(
        .clk(clk), .rst_n(rst_n),
        .start(sage16_start),
        .a_in(a_flat), .b_in(b_flat),
        .c_out(c_flat), .done(sage16_done)
    );

    // ----- busy tracking + irq -----
    reg        busy;
    reg        done_r, done_r_d;
    assign     irq = done_r & ~done_r_d;

    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            sage16_start <= 0;
            busy       <= 0;
            done_r     <= 0;
            done_r_d   <= 0;
            ack        <= 0;
            rdata      <= 32'd0;
            for(ii=0; ii<16; ii=ii+1) begin
                a_buf[ii] <= 0;
                b_buf[ii] <= 0;
            end
        end else begin
            sage16_start <= 0;      // default: no start
            ack        <= we | re;
            done_r_d   <= done_r;
            done_r     <= sage16_done;

            // ---- writes ----
            if(is_A_write) a_buf[ab_idx] <= wdata[DATA_W-1:0];
            if(is_B_write) b_buf[ab_idx] <= wdata[DATA_W-1:0];
            if(is_CTRL_wr && wdata[0] && !busy) begin
                sage16_start <= 1;
                busy       <= 1;
            end
            if(sage16_done) busy <= 0;

            // ---- reads ----
            rdata <= 32'd0;
            if(is_C_read)
                rdata <= c_flat[c_idx*ACC_W +: ACC_W];
            else if(is_STATUS_rd)
                rdata <= {30'd0, busy, sage16_done};
        end
    end
endmodule
