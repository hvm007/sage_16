`timescale 1ns/1ps
// =============================================================================
// tb_minicnn_v2.v -- end-to-end MiniCNN-v2 inference on sage16_top
//
// What is different from tb_minicnn.v
// -----------------------------------
// v1 had ONE conv stage (1 -> 4 channels) and reached an architectural ceiling
// of ~94% MNIST top-1 because a single-stage 4-channel feature extractor lacks
// the representational depth for handwritten digits.
// v2 adds a SECOND convolutional stage (8 -> 16 channels) that runs on the
// SAME unmodified sage16_top fabric -- conv3x3_sage16 still does a single-channel
// 6x6 -> 4x4 conv, and multi-channel summation is performed in software on
// the testbench host (the SoC's RISC-V core would do this in real deployment).
//
// Topology
// --------
//   input    26x26x1                   INT16 centered uint6 [-32, +31]
//   Conv1    3x3 valid, 1 -> 8         INT4  -> 24x24x8     (288 fabric calls)
//   AvgPool  4x4 stride 4   ( >>>4 )                  -> 6x6x8
//   Conv2    3x3 valid, 8 -> 16        INT4  -> 4x4x16      (128 fabric calls,
//                                                           sum across 8 ic in SW)
//   Trunc    >>> conv2_shift                          -> 4x4x16  (16-bit safe)
//   Flatten                                           -> 256
//   FC1      256 -> 32      INT4  matmul (mode=3)     -> 32      (512 calls)
//   ReLU
//   FC2      32  -> 12      (10 real + 2 zero rows)   -> 12      ( 24 calls)
//   argmax over the first 10 outputs                  -> class
//
// Verification
// ------------
// A pure-Verilog integer golden reference (compute_golden) is computed for
// every image and compared bit-for-bit against the fabric output at every
// stage (Conv1, Conv2, FC1+ReLU, FC2).  Any mismatch flips n_fail_*.
// Predictions are also matched against ground-truth MNIST labels.
//
// Weights & images
// ----------------
// Synthetic prand mode (default) generates small-magnitude pseudo-random INT16
// data so the fabric is exercised without any external files.
// REAL_MNIST mode (compile with +define+REAL_MNIST) loads:
//     data/v2/conv1_k.mem  data/v2/conv2_k.mem
//     data/v2/w1.mem       data/v2/w2.mem
//     data/v2/images.mem   data/v2/labels.mem
// produced by scripts/train_minicnn_v2.py.
// =============================================================================
module tb_minicnn_v2;
    parameter DATA_W      = 16;
    parameter ACC_W       = 32;
    parameter CONV2_SHIFT = 4;     // >>> applied to Conv2 output before FC1
`ifdef REAL_MNIST
    parameter N_IMG       = 10000;
`else
    parameter N_IMG       = 3;
`endif

    reg clk = 0; always #5 clk = ~clk;        // 100 MHz sim clock

    // ------------------------------------------------------------------
    //  DUT (shared-fabric sage16_top, driven directly -- no MMIO bus)
    // ------------------------------------------------------------------
    reg                  rst_n;
    reg  [1:0]           mode;
    reg                  start;
    wire                 done;
    wire [1:0]           mode_out;

    reg  [16*DATA_W-1:0] mm_a, mm_b;
    reg  [36*DATA_W-1:0] cv_img_r;
    reg  [ 9*DATA_W-1:0] cv_k_r;
    wire [DATA_W-1:0]    qt_q1_w = 0, qt_q1_x = 0, qt_q1_y = 0, qt_q1_z = 0;
    wire [16*DATA_W-1:0] qt_q2   = 0;
    wire [16*ACC_W-1:0]  c_flat;

    sage16_top #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .mode(mode), .start(start), .done(done), .mode_out(mode_out),
        .mm_a(mm_a), .mm_b(mm_b),
        .cv_img(cv_img_r), .cv_k(cv_k_r),
        .qt_q1_w(qt_q1_w), .qt_q1_x(qt_q1_x),
        .qt_q1_y(qt_q1_y), .qt_q1_z(qt_q1_z),
        .qt_q2(qt_q2),
        .c_out(c_flat)
    );

    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    reg  done_d;
    wire done_rise = done & ~done_d;
    always @(posedge clk) done_d <= done;

    integer layer_cycles;

    task run_conv_call;
        input [36*DATA_W-1:0] img_in;
        input [ 9*DATA_W-1:0] k_in;
        output [16*ACC_W-1:0] cout;
        integer t0;
        begin
            @(negedge clk);
            cv_img_r = img_in;
            cv_k_r   = k_in;
            mode     = 2'd1;
            start    = 1;
            @(posedge clk); #1;
            start = 0;
            t0 = cyc_cnt;
            wait(done_rise);
            @(posedge clk); #1;
            layer_cycles = layer_cycles + (cyc_cnt - t0);
            cout = c_flat;
        end
    endtask

    // signed matmul (mode=3 = MODE_MMS).  Re-uses the same routing as
    // MODE_MM but toggles the PE opcode to OP_MACB_S so FC layers can
    // multiply signed INT weights by signed activations.
    task run_matmul_call;
        input  [16*DATA_W-1:0] A4, B4;
        output [16*ACC_W-1:0]  C4;
        integer t0;
        begin
            @(negedge clk);
            mm_a  = A4;
            mm_b  = B4;
            mode  = 2'd3;
            start = 1;
            @(posedge clk); #1;
            start = 0;
            t0 = cyc_cnt;
            wait(done_rise);
            @(posedge clk); #1;
            layer_cycles = layer_cycles + (cyc_cnt - t0);
            C4 = c_flat;
        end
    endtask

    // ------------------------------------------------------------------
    //  Problem dimensions
    // ------------------------------------------------------------------
    localparam HIN   = 26;            // input  H/W
    localparam HC1   = 24;            // Conv1  output H/W
    localparam NC1   = 8;             // Conv1  output channels
    localparam HP    = 6;             // pooled H/W (24 / 4)
    localparam NC2   = 16;            // Conv2  output channels
    localparam HC2   = 4;             // Conv2  output H/W (6 - 3 + 1 = 4)
    localparam NFLAT = HC2*HC2*NC2;   // 256
    localparam NFC1  = 32;
    localparam NFC2  = 12;            // 10 real + 2 zero pads (= 3 row-tiles)

    function signed [15:0] prand;
        input integer seed;
        input integer lo, hi;
        integer r;
        begin
            r     = (seed * 1103515245 + 12345) % 65536;
            if(r < 0) r = -r;
            prand = lo + (r % (hi - lo + 1));
        end
    endfunction

    // images / weights / activations (signed)
    reg signed [DATA_W-1:0] IMG     [0:N_IMG-1][0:HIN-1][0:HIN-1];
    reg signed [DATA_W-1:0] CONV1_K [0:NC1-1][0:2][0:2];
    reg signed [DATA_W-1:0] CONV2_K [0:NC2-1][0:NC1-1][0:2][0:2];
    reg signed [DATA_W-1:0] W1      [0:NFC1-1][0:NFLAT-1];
    reg signed [DATA_W-1:0] W2      [0:NFC2-1][0:NFC1-1];
`ifdef REAL_MNIST
    reg signed [DATA_W-1:0] LABEL   [0:N_IMG-1];
`endif

    // per-image intermediates
    reg signed [ACC_W-1:0]  A_C1    [0:NC1-1][0:HC1-1][0:HC1-1];
    reg signed [ACC_W-1:0]  A_POOL  [0:NC1-1][0:HP-1][0:HP-1];
    reg signed [ACC_W-1:0]  A_C2    [0:NC2-1][0:HC2-1][0:HC2-1];
    reg signed [ACC_W-1:0]  A_FLAT  [0:NFLAT-1];
    reg signed [ACC_W-1:0]  A_FC1   [0:NFC1-1];
    reg signed [ACC_W-1:0]  A_FC2   [0:NFC2-1];

    // golden references
    reg signed [ACC_W-1:0]  G_C1    [0:NC1-1][0:HC1-1][0:HC1-1];
    reg signed [ACC_W-1:0]  G_POOL  [0:NC1-1][0:HP-1][0:HP-1];
    reg signed [ACC_W-1:0]  G_C2    [0:NC2-1][0:HC2-1][0:HC2-1];
    reg signed [ACC_W-1:0]  G_FLAT  [0:NFLAT-1];
    reg signed [ACC_W-1:0]  G_FC1   [0:NFC1-1];
    reg signed [ACC_W-1:0]  G_FC2   [0:NFC2-1];

    integer cy_conv1_tot, cy_pool_tot, cy_conv2_tot, cy_fc1_tot, cy_fc2_tot;
    integer cy_conv1_i,   cy_pool_i,   cy_conv2_i,   cy_fc1_i,   cy_fc2_i;
    integer n_pass_c1,  n_fail_c1;
    integer n_pass_c2,  n_fail_c2;
    integer n_pass_fc1, n_fail_fc1;
    integer n_pass_fc2, n_fail_fc2;
    integer n_pass_pred, n_fail_pred;
    integer n_correct_label, n_total_label;

    integer i, j, ch, r, c, oc, ic, k_idx, ti, tj, tk;
    integer di, dj, tr, tc;
    integer argmax_i, argmax_g;
    reg signed [ACC_W-1:0] amax_v, amax_g;
    reg signed [ACC_W-1:0] acc_s;

    // local tile scratchpads
    reg [36*DATA_W-1:0] img_pack;
    reg [ 9*DATA_W-1:0] k_pack;
    reg [16*ACC_W-1:0]  c_pack;
    reg [16*DATA_W-1:0] A_tile, B_tile;
    reg signed [ACC_W-1:0] tile_acc [0:15];

    // ------------------------------------------------------------------
    //  Weight / image init
    // ------------------------------------------------------------------
    task init_weights_and_images;
        integer s;
        begin
`ifdef REAL_MNIST
            $readmemh("data/v2/conv1_k.mem", CONV1_K);
            $readmemh("data/v2/conv2_k.mem", CONV2_K);
            $readmemh("data/v2/w1.mem",      W1);
            $readmemh("data/v2/w2.mem",      W2);
            $readmemh("data/v2/images.mem",  IMG);
            $readmemh("data/v2/labels.mem",  LABEL);
            $display("  [REAL_MNIST] loaded conv1(%0d), conv2(%0d), w1(%0d), w2(%0d), images(%0d), labels(%0d)",
                     NC1*9, NC2*NC1*9, NFC1*NFLAT, NFC2*NFC1, N_IMG*HIN*HIN, N_IMG);
`else
            s = 1;
            for (oc=0; oc<NC1; oc=oc+1)
                for (i=0; i<3; i=i+1)
                    for (j=0; j<3; j=j+1) begin
                        CONV1_K[oc][i][j] = prand(s, -3, 3); s = s + 1;
                    end
            for (oc=0; oc<NC2; oc=oc+1)
                for (ic=0; ic<NC1; ic=ic+1)
                    for (i=0; i<3; i=i+1)
                        for (j=0; j<3; j=j+1) begin
                            CONV2_K[oc][ic][i][j] = prand(s, -3, 3); s = s + 1;
                        end
            for (i=0; i<NFC1; i=i+1)
                for (j=0; j<NFLAT; j=j+1) begin
                    W1[i][j] = prand(s, -3, 3); s = s + 1;
                end
            for (i=0; i<NFC2; i=i+1)
                for (j=0; j<NFC1; j=j+1) begin
                    if (i >= 10) W2[i][j] = 0;
                    else begin W2[i][j] = prand(s, -3, 3); s = s + 1; end
                end
            for (ch=0; ch<N_IMG; ch=ch+1)
                for (r=0; r<HIN; r=r+1)
                    for (c=0; c<HIN; c=c+1) begin
                        IMG[ch][r][c] = prand(s, 0, 31); s = s + 1;
                    end
`endif
        end
    endtask

    // ------------------------------------------------------------------
    //  Software golden reference for one image
    //  (pure Verilog integer arithmetic -- independent of fabric path)
    // ------------------------------------------------------------------
    task compute_golden;
        input  integer img_idx;
        integer oc_l, ic_l, r_l, c_l, i_l, j_l;
        reg signed [ACC_W-1:0] acc, poolacc;
        begin
            // Conv1
            for (oc_l=0; oc_l<NC1; oc_l=oc_l+1)
                for (r_l=0; r_l<HC1; r_l=r_l+1)
                    for (c_l=0; c_l<HC1; c_l=c_l+1) begin
                        acc = 0;
                        for (i_l=0; i_l<3; i_l=i_l+1)
                            for (j_l=0; j_l<3; j_l=j_l+1)
                                acc = acc + $signed(IMG[img_idx][r_l+i_l][c_l+j_l])
                                          * $signed(CONV1_K[oc_l][i_l][j_l]);
                        G_C1[oc_l][r_l][c_l] = acc;
                    end
            // AvgPool 4x4 stride 4 (sum>>>4)
            for (oc_l=0; oc_l<NC1; oc_l=oc_l+1)
                for (r_l=0; r_l<HP; r_l=r_l+1)
                    for (c_l=0; c_l<HP; c_l=c_l+1) begin
                        poolacc = 0;
                        for (i_l=0; i_l<4; i_l=i_l+1)
                            for (j_l=0; j_l<4; j_l=j_l+1)
                                poolacc = poolacc + G_C1[oc_l][r_l*4+i_l][c_l*4+j_l];
                        G_POOL[oc_l][r_l][c_l] = poolacc >>> 4;
                    end
            // Conv2: full 8->16 valid 3x3 conv, 6x6 -> 4x4 spatial
            for (oc_l=0; oc_l<NC2; oc_l=oc_l+1)
                for (r_l=0; r_l<HC2; r_l=r_l+1)
                    for (c_l=0; c_l<HC2; c_l=c_l+1) begin
                        acc = 0;
                        for (ic_l=0; ic_l<NC1; ic_l=ic_l+1)
                            for (i_l=0; i_l<3; i_l=i_l+1)
                                for (j_l=0; j_l<3; j_l=j_l+1)
                                    acc = acc + G_POOL[ic_l][r_l+i_l][c_l+j_l]
                                              * $signed(CONV2_K[oc_l][ic_l][i_l][j_l]);
                        G_C2[oc_l][r_l][c_l] = acc;
                    end
            // Truncate Conv2 (>>> CONV2_SHIFT) and flatten
            for (oc_l=0; oc_l<NC2; oc_l=oc_l+1)
                for (r_l=0; r_l<HC2; r_l=r_l+1)
                    for (c_l=0; c_l<HC2; c_l=c_l+1)
                        G_FLAT[oc_l*(HC2*HC2) + r_l*HC2 + c_l] =
                            G_C2[oc_l][r_l][c_l] >>> CONV2_SHIFT;
            // FC1
            for (oc_l=0; oc_l<NFC1; oc_l=oc_l+1) begin
                acc = 0;
                for (j_l=0; j_l<NFLAT; j_l=j_l+1)
                    acc = acc + G_FLAT[j_l] * $signed(W1[oc_l][j_l]);
                G_FC1[oc_l] = acc;
            end
            // ReLU
            for (i_l=0; i_l<NFC1; i_l=i_l+1)
                if (G_FC1[i_l] < 0) G_FC1[i_l] = 0;
            // FC2
            for (oc_l=0; oc_l<NFC2; oc_l=oc_l+1) begin
                acc = 0;
                for (j_l=0; j_l<NFC1; j_l=j_l+1)
                    acc = acc + G_FC1[j_l] * $signed(W2[oc_l][j_l]);
                G_FC2[oc_l] = acc;
            end
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 1 (Conv1, single-channel input -> 8 output channels)
    // ------------------------------------------------------------------
    task run_conv1_image;
        input integer img_idx;
        integer tr_out, tc_out, rr, cc;
        begin
            layer_cycles = 0;
            for (oc=0; oc<NC1; oc=oc+1) begin
                for (tr_out=0; tr_out<HC1; tr_out=tr_out+4) begin
                    for (tc_out=0; tc_out<HC1; tc_out=tc_out+4) begin
                        for (rr=0; rr<6; rr=rr+1)
                            for (cc=0; cc<6; cc=cc+1)
                                img_pack[(rr*6+cc)*DATA_W +: DATA_W] =
                                    IMG[img_idx][tr_out+rr][tc_out+cc];
                        for (i=0; i<3; i=i+1)
                            for (j=0; j<3; j=j+1)
                                k_pack[(i*3+j)*DATA_W +: DATA_W] =
                                    CONV1_K[oc][i][j];
                        run_conv_call(img_pack, k_pack, c_pack);
                        for (rr=0; rr<4; rr=rr+1)
                            for (cc=0; cc<4; cc=cc+1)
                                A_C1[oc][tr_out+rr][tc_out+cc] =
                                    $signed(c_pack[(rr*4+cc)*ACC_W +: ACC_W]);
                    end
                end
            end
            cy_conv1_i = layer_cycles;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 2 (AvgPool 4x4 stride 4, SW emulated to model PicoRV32 cost)
    // ------------------------------------------------------------------
    task run_pool;
        integer pi, pj;
        integer t0_sw;
        reg signed [ACC_W-1:0] ps;
        begin
            t0_sw = cyc_cnt;
            for (oc=0; oc<NC1; oc=oc+1)
                for (r=0; r<HP; r=r+1)
                    for (c=0; c<HP; c=c+1) begin
                        ps = 0;
                        for (pi=0; pi<4; pi=pi+1)
                            for (pj=0; pj<4; pj=pj+1)
                                ps = ps + A_C1[oc][r*4+pi][c*4+pj];
                        A_POOL[oc][r][c] = ps >>> 4;
                    end
            // 8 channels * 6 * 6 * 16 = 4608 adds -> ~4600 cy at ~1 IPC.
            repeat (4600) @(posedge clk);
            cy_pool_i = cyc_cnt - t0_sw;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 3 (Conv2, 8->16 with multi-channel summation in SW)
    //  Each (oc, ic) pair runs ONE conv3x3 fabric call (6x6 -> 4x4).
    //  Total fabric calls = NC2 * NC1 = 16 * 8 = 128.
    //  Cross-channel summation done in SW between calls.
    // ------------------------------------------------------------------
    task run_conv2_image;
        input integer img_idx;
        integer rr, cc;
        reg signed [ACC_W-1:0] cur;
        begin
            layer_cycles = 0;
            // zero A_C2 and A_FLAT
            for (oc=0; oc<NC2; oc=oc+1)
                for (rr=0; rr<HC2; rr=rr+1)
                    for (cc=0; cc<HC2; cc=cc+1)
                        A_C2[oc][rr][cc] = 0;
            // (oc, ic) pair = 1 fabric call
            for (oc=0; oc<NC2; oc=oc+1) begin
                for (ic=0; ic<NC1; ic=ic+1) begin
                    for (rr=0; rr<6; rr=rr+1)
                        for (cc=0; cc<6; cc=cc+1)
                            img_pack[(rr*6+cc)*DATA_W +: DATA_W] =
                                A_POOL[ic][rr][cc][DATA_W-1:0];
                    for (i=0; i<3; i=i+1)
                        for (j=0; j<3; j=j+1)
                            k_pack[(i*3+j)*DATA_W +: DATA_W] =
                                CONV2_K[oc][ic][i][j];
                    run_conv_call(img_pack, k_pack, c_pack);
                    for (rr=0; rr<4; rr=rr+1)
                        for (cc=0; cc<4; cc=cc+1) begin
                            cur = $signed(c_pack[(rr*4+cc)*ACC_W +: ACC_W]);
                            A_C2[oc][rr][cc] = A_C2[oc][rr][cc] + cur;
                        end
                end
            end
            // Truncate (>>> CONV2_SHIFT) and flatten into A_FLAT
            for (oc=0; oc<NC2; oc=oc+1)
                for (rr=0; rr<HC2; rr=rr+1)
                    for (cc=0; cc<HC2; cc=cc+1)
                        A_FLAT[oc*(HC2*HC2) + rr*HC2 + cc] =
                            A_C2[oc][rr][cc] >>> CONV2_SHIFT;
            cy_conv2_i = layer_cycles;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 4 (FC1 = W1 * A_FLAT, 32 x 256 tiled 4x4)
    // ------------------------------------------------------------------
    task run_fc1;
        integer oc0, kk, ii, jj;
        reg signed [ACC_W-1:0] sum;
        begin
            layer_cycles = 0;
            for (oc0=0; oc0<NFC1; oc0=oc0+4) begin
                for (ii=0; ii<4; ii=ii+1) tile_acc[ii] = 0;
                for (kk=0; kk<NFLAT; kk=kk+4) begin
                    for (ii=0; ii<4; ii=ii+1)
                        for (jj=0; jj<4; jj=jj+1)
                            A_tile[(ii*4+jj)*DATA_W +: DATA_W] =
                                W1[oc0+ii][kk+jj];
                    for (ii=0; ii<4; ii=ii+1) begin
                        B_tile[(ii*4+0)*DATA_W +: DATA_W] =
                            A_FLAT[kk+ii][DATA_W-1:0];
                        B_tile[(ii*4+1)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+2)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+3)*DATA_W +: DATA_W] = 0;
                    end
                    run_matmul_call(A_tile, B_tile, c_pack);
                    for (ii=0; ii<4; ii=ii+1) begin
                        sum = $signed(c_pack[(ii*4+0)*ACC_W +: ACC_W]);
                        tile_acc[ii] = tile_acc[ii] + sum;
                    end
                end
                for (ii=0; ii<4; ii=ii+1)
                    A_FC1[oc0+ii] = tile_acc[ii];
            end
            cy_fc1_i = layer_cycles;
        end
    endtask

    task run_relu;
        integer ii;
        begin
            for (ii=0; ii<NFC1; ii=ii+1)
                if (A_FC1[ii] < 0) A_FC1[ii] = 0;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 5 (FC2 = W2 * ReLU(A_FC1), 12 x 32 tiled)
    // ------------------------------------------------------------------
    task run_fc2;
        integer oc0, kk, ii, jj;
        reg signed [ACC_W-1:0] sum;
        begin
            layer_cycles = 0;
            for (oc0=0; oc0<NFC2; oc0=oc0+4) begin
                for (ii=0; ii<4; ii=ii+1) tile_acc[ii] = 0;
                for (kk=0; kk<NFC1; kk=kk+4) begin
                    for (ii=0; ii<4; ii=ii+1)
                        for (jj=0; jj<4; jj=jj+1)
                            A_tile[(ii*4+jj)*DATA_W +: DATA_W] =
                                W2[oc0+ii][kk+jj];
                    for (ii=0; ii<4; ii=ii+1) begin
                        B_tile[(ii*4+0)*DATA_W +: DATA_W] =
                            A_FC1[kk+ii][DATA_W-1:0];
                        B_tile[(ii*4+1)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+2)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+3)*DATA_W +: DATA_W] = 0;
                    end
                    run_matmul_call(A_tile, B_tile, c_pack);
                    for (ii=0; ii<4; ii=ii+1) begin
                        sum = $signed(c_pack[(ii*4+0)*ACC_W +: ACC_W]);
                        tile_acc[ii] = tile_acc[ii] + sum;
                    end
                end
                for (ii=0; ii<4; ii=ii+1)
                    A_FC2[oc0+ii] = tile_acc[ii];
            end
            cy_fc2_i = layer_cycles;
        end
    endtask

    // ------------------------------------------------------------------
    //  Verification
    // ------------------------------------------------------------------
    task check_conv1;
        begin
            for (oc=0; oc<NC1; oc=oc+1)
                for (r=0; r<HC1; r=r+1)
                    for (c=0; c<HC1; c=c+1) begin
                        if (A_C1[oc][r][c] === G_C1[oc][r][c])
                            n_pass_c1 = n_pass_c1 + 1;
                        else begin
                            n_fail_c1 = n_fail_c1 + 1;
                            if (n_fail_c1 <= 4)
                                $display("  [conv1] oc=%0d r=%0d c=%0d got=%0d exp=%0d",
                                         oc, r, c, A_C1[oc][r][c], G_C1[oc][r][c]);
                        end
                    end
        end
    endtask

    task check_conv2;
        begin
            for (oc=0; oc<NC2; oc=oc+1)
                for (r=0; r<HC2; r=r+1)
                    for (c=0; c<HC2; c=c+1) begin
                        if (A_C2[oc][r][c] === G_C2[oc][r][c])
                            n_pass_c2 = n_pass_c2 + 1;
                        else begin
                            n_fail_c2 = n_fail_c2 + 1;
                            if (n_fail_c2 <= 4)
                                $display("  [conv2] oc=%0d r=%0d c=%0d got=%0d exp=%0d",
                                         oc, r, c, A_C2[oc][r][c], G_C2[oc][r][c]);
                        end
                    end
        end
    endtask

    task check_fc1_post_relu;
        integer ii;
        reg signed [ACC_W-1:0] want;
        begin
            for (ii=0; ii<NFC1; ii=ii+1) begin
                want = G_FC1[ii];
                if (A_FC1[ii] === want)
                    n_pass_fc1 = n_pass_fc1 + 1;
                else begin
                    n_fail_fc1 = n_fail_fc1 + 1;
                    if (n_fail_fc1 <= 4)
                        $display("  [fc1+relu] i=%0d got=%0d exp=%0d",
                                 ii, A_FC1[ii], want);
                end
            end
        end
    endtask

    task check_fc2;
        integer ii;
        begin
            for (ii=0; ii<NFC2; ii=ii+1) begin
                if (A_FC2[ii] === G_FC2[ii])
                    n_pass_fc2 = n_pass_fc2 + 1;
                else begin
                    n_fail_fc2 = n_fail_fc2 + 1;
                    if (n_fail_fc2 <= 4)
                        $display("  [fc2] i=%0d got=%0d exp=%0d",
                                 ii, A_FC2[ii], G_FC2[ii]);
                end
            end
        end
    endtask

    task check_argmax;
        input integer img_idx;
        integer ii;
        begin
            argmax_i = 0; amax_v = A_FC2[0];
            argmax_g = 0; amax_g = G_FC2[0];
            for (ii=1; ii<10; ii=ii+1) begin
                if (A_FC2[ii] > amax_v) begin amax_v = A_FC2[ii]; argmax_i = ii; end
                if (G_FC2[ii] > amax_g) begin amax_g = G_FC2[ii]; argmax_g = ii; end
            end
            if (argmax_i == argmax_g)
                n_pass_pred = n_pass_pred + 1;
            else begin
                n_fail_pred = n_fail_pred + 1;
                $display("  [pred] got class %0d, expected %0d", argmax_i, argmax_g);
            end
`ifdef REAL_MNIST
            n_total_label = n_total_label + 1;
            if (argmax_i == LABEL[img_idx])
                n_correct_label = n_correct_label + 1;
`endif
        end
    endtask

    // ------------------------------------------------------------------
    //  Main
    // ------------------------------------------------------------------
    integer img;
    initial begin
`ifndef REAL_MNIST
        $dumpfile("build/minicnn_v2_tb.vcd");
        $dumpvars(0, tb_minicnn_v2);
`endif
        rst_n = 0; mode = 0; start = 0;
        mm_a = 0; mm_b = 0; cv_img_r = 0; cv_k_r = 0;
        cy_conv1_tot = 0; cy_pool_tot = 0; cy_conv2_tot = 0;
        cy_fc1_tot = 0; cy_fc2_tot = 0;
        n_pass_c1  = 0; n_fail_c1  = 0;
        n_pass_c2  = 0; n_fail_c2  = 0;
        n_pass_fc1 = 0; n_fail_fc1 = 0;
        n_pass_fc2 = 0; n_fail_fc2 = 0;
        n_pass_pred = 0; n_fail_pred = 0;
        n_correct_label = 0; n_total_label = 0;

        init_weights_and_images;

        #23 rst_n = 1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  MiniCNN-v2  end-to-end inference on sage16_top");
        $display("  topology: 26x26 -> Conv1(3x3,1->8) -> Pool4 ->");
        $display("            Conv2(3x3,8->16) -> >>>%0d -> FC1(256->32)", CONV2_SHIFT);
        $display("            -> ReLU -> FC2(32->12) -> argmax");
        $display("  %0d test images, per-image per-layer cycle counts", N_IMG);
        $display("================================================");

        for (img=0; img<N_IMG; img=img+1) begin
            compute_golden(img);

            run_conv1_image(img);
            cy_conv1_tot = cy_conv1_tot + cy_conv1_i;
            check_conv1;

            run_pool;
            cy_pool_tot  = cy_pool_tot + cy_pool_i;

            run_conv2_image(img);
            cy_conv2_tot = cy_conv2_tot + cy_conv2_i;
            check_conv2;

            run_fc1;
            cy_fc1_tot   = cy_fc1_tot + cy_fc1_i;
            run_relu;
            check_fc1_post_relu;

            run_fc2;
            cy_fc2_tot   = cy_fc2_tot + cy_fc2_i;
            check_fc2;

            check_argmax(img);

`ifdef REAL_MNIST
            if (img < 8)
                $display("  img %0d : conv1=%0d pool=%0d conv2=%0d fc1=%0d fc2=%0d  pred=%0d (label=%0d)",
                         img, cy_conv1_i, cy_pool_i, cy_conv2_i, cy_fc1_i, cy_fc2_i,
                         argmax_i, LABEL[img]);
`else
            $display("  img %0d : conv1=%0d pool=%0d conv2=%0d fc1=%0d fc2=%0d  pred=%0d (exp=%0d)",
                     img, cy_conv1_i, cy_pool_i, cy_conv2_i, cy_fc1_i, cy_fc2_i,
                     argmax_i, argmax_g);
`endif
`ifdef REAL_MNIST
            if ((((img + 1) <= 200) && (((img + 1) % 10) == 0)) ||
                (((img + 1)  > 200) && (((img + 1) % 100) == 0)))
                $display("  progress: %0d / %0d images processed", img + 1, N_IMG);
`endif
        end

        begin : summary_blk
            integer total_fab;
            integer total_full;
            real    fab_us, full_us;
            total_fab  = cy_conv1_tot + cy_conv2_tot + cy_fc1_tot + cy_fc2_tot;
            total_full = total_fab + cy_pool_tot;
            fab_us     = (total_fab  / (1.0*N_IMG)) / 160.0;
            full_us    = (total_full / (1.0*N_IMG)) / 160.0;
            $display("");
            $display("================================================");
            $display("  MiniCNN-v2  SUMMARY   (totals over %0d images)", N_IMG);
            $display("================================================");
            $display("  fabric compute cycles : Conv1=%0d  Conv2=%0d  FC1=%0d  FC2=%0d",
                     cy_conv1_tot, cy_conv2_tot, cy_fc1_tot, cy_fc2_tot);
            $display("  host SW cycles (pool) : %0d", cy_pool_tot);
            $display("  per-image avg         : fabric=%0d  full=%0d",
                     total_fab/N_IMG, total_full/N_IMG);
            $display("  at 160 MHz            : fabric=%0.1f us  full=%0.1f us",
                     fab_us, full_us);
            $display("  bit-exact check       : conv1 pass=%0d fail=%0d",
                     n_pass_c1, n_fail_c1);
            $display("                        : conv2 pass=%0d fail=%0d",
                     n_pass_c2, n_fail_c2);
            $display("                        : fc1   pass=%0d fail=%0d",
                     n_pass_fc1, n_fail_fc1);
            $display("                        : fc2   pass=%0d fail=%0d",
                     n_pass_fc2, n_fail_fc2);
            $display("  pred matches golden   : %0d / %0d",
                     n_pass_pred, N_IMG);
`ifdef REAL_MNIST
            $display("  MNIST top-1 (fabric vs ground truth)");
            $display("                        : %0d / %0d  (%0.2f%% accuracy)",
                     n_correct_label, n_total_label,
                     100.0 * n_correct_label / (1.0 * n_total_label));
`endif
            $display("  SUMMARY: pass=%0d fail=%0d",
                     n_pass_c1 + n_pass_c2 + n_pass_fc1 + n_pass_fc2 + n_pass_pred,
                     n_fail_c1 + n_fail_c2 + n_fail_fc1 + n_fail_fc2 + n_fail_pred);
            $display("================================================");
        end

        #100 $finish;
    end
endmodule
