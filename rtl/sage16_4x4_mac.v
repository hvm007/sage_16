// =============================================================================
// sage16_4x4_mac.v  — 4x4 sage16 fabric
//
// PE array with:
//   - Per-row A bypass network          (combinational broadcast to row i)
//   - Per-column B broadcast network    (combinational broadcast to col j)
//   - Mesh N/S/E/W interconnect         (for non-MACB ops)
//   - 1-cycle parallel config-broadcast (all PEs program in 1 clk)
//   - Parallel accumulator clear         (all PEs zero in 1 clk)
//   - Flattened full-width readout       (all 16 PE ACC_W outputs exposed)
//
// Scalability note (for the paper):
//   The broadcast networks are combinational. For a 4x4 fabric the fan-out
//   is 4, which is routable at >200 MHz on Artix-7 with DSP pipelining.
//   For >=16x16 fabrics, the broadcast should be pipelined (1 register per
//   row/column hop) — that's a drop-in change by making bypass/b_col
//   registered in a generate loop. Tracked as future work.
// =============================================================================
module sage16_4x4_mac #(
    parameter ROWS     = 4,
    parameter COLS     = 4,
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1
)(
    input                             clk, rst_n,
    input      [$clog2(ROWS)-1:0]     cfg_pe_row,
    input      [$clog2(COLS)-1:0]     cfg_pe_col,
    input      [CFG_W-1:0]            cfg_data,
    input                             cfg_load,
    input                             cfg_broadcast,
    input                             clr_acc_all,
    input                             out_en_all,
    input      [ROWS*DATA_W-1:0]      ext_in_west,   // A[i][k] row-broadcast
    input      [COLS*DATA_W-1:0]      ext_in_north,  // B[k][j] col-broadcast
    // ------------------------------------------------------------
    // Per-PE bypass path (used by conv3x3 which needs per-PE pixels
    // that cannot be row-broadcast). When per_pe_bypass_en=0, the
    // classical row-broadcast is used and this fabric is exactly the
    // original matmul fabric — backward compatible.
    // ------------------------------------------------------------
    input                             per_pe_bypass_en,
    input      [ROWS*COLS*DATA_W-1:0] per_pe_bypass_flat,
    output     [ROWS*ACC_W-1:0]       ext_out_east,
    output     [ROWS*COLS*ACC_W-1:0]  all_pe_out
);
    wire [ACC_W-1:0]  pe_out_acc  [0:ROWS*COLS-1];
    wire [DATA_W-1:0] pe_out_mesh [0:ROWS*COLS-1];

    genvar r, c;
    generate for(r=0; r<ROWS; r=r+1) begin : rg
        for(c=0; c<COLS; c=c+1) begin : cg
            wire cfl = cfg_broadcast ||
                       (cfg_load && (cfg_pe_row==r) && (cfg_pe_col==c));

            wire [DATA_W-1:0] wn, ws, we, ww;
            if(r==0)       assign wn = ext_in_north[c*DATA_W +: DATA_W];
            else           assign wn = pe_out_mesh[(r-1)*COLS + c];
            if(r==ROWS-1)  assign ws = {DATA_W{1'b0}};
            else           assign ws = pe_out_mesh[(r+1)*COLS + c];
            if(c==COLS-1)  assign we = {DATA_W{1'b0}};
            else           assign we = pe_out_mesh[r*COLS + (c+1)];
            if(c==0)       assign ww = ext_in_west[r*DATA_W +: DATA_W];
            else           assign ww = pe_out_mesh[r*COLS + (c-1)];

            wire [DATA_W-1:0] bypass_bcast = ext_in_west [r*DATA_W +: DATA_W];
            wire [DATA_W-1:0] bypass_perpe =
                per_pe_bypass_flat[(r*COLS + c)*DATA_W +: DATA_W];
            wire [DATA_W-1:0] bypass = per_pe_bypass_en ? bypass_perpe
                                                        : bypass_bcast;
            wire [DATA_W-1:0] b_col  = ext_in_north[c*DATA_W +: DATA_W];

            pe #(
                .DATA_W  (DATA_W),
                .ACC_W   (ACC_W),
                .CFG_W   (CFG_W),
                .PIPELINE(PIPELINE)
            ) u_pe(
                .clk(clk), .rst_n(rst_n),
                .cfg_load(cfl), .cfg_in(cfg_data),
                .out_en(out_en_all),
                .clr_acc(clr_acc_all),
                .in_north(wn), .in_south(ws), .in_east(we), .in_west(ww),
                .in_self (pe_out_acc [r*COLS + c]),
                .in_bypass(bypass), .in_b_col(b_col),
                .out_mesh(pe_out_mesh[r*COLS + c]),
                .out     (pe_out_acc [r*COLS + c])
            );

            if(c==COLS-1)
                assign ext_out_east[r*ACC_W +: ACC_W] = pe_out_acc[r*COLS + c];

            assign all_pe_out[(r*COLS + c)*ACC_W +: ACC_W] = pe_out_acc[r*COLS + c];
        end
    end endgenerate
endmodule
