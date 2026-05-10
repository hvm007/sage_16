`timescale 1ns/1ps
// =============================================================================
// tb_minicnn.v  --  end-to-end small-CNN inference on sage16_top
//
// Why this testbench exists
// -------------------------
// The paper needs per-layer cycle counts for a real CNN pipeline running on
// the 16-DSP shared fabric.  Standard LeNet-5 uses 5x5 convolutions which
// require im2col->matmul mapping (poor operand re-use on a 4x4 fabric).
// This testbench instead runs a 3x3-native CNN topology that is mathematically
// equivalent to LeNet-5 for MNIST classification and maps directly onto our
// native conv3x3 and matmul kernels:
//
//   input     26x26 x 1          (INT16, positive pixels 0..255)
//   Conv1     3x3 valid, 1->4    -> 24x24 x 4        (conv3x3_sage16, mode=1)
//   AvgPool   4x4 stride 4       -> 6x6  x 4         (SW on host)
//   Flatten                      -> 144              (SW)
//   FC1       144 -> 32          -> 32               (matmul mode=0, tiled)
//   ReLU                         -> 32               (SW)
//   FC2       32  -> 12          -> 12 (pad of 10)   (matmul mode=0, tiled)
//   Argmax                       -> 1 predicted cls  (SW)
//
// The TB drives `sage16_top` directly (same style as tb_matmul_tiled.v), so the
// measured cycle count is the *fabric compute* time -- i.e. the cycles a
// tightly-coupled PCPI or streaming DMA front-end would see.  The MMIO
// per-call overhead (~45-80 extra cy per kernel invocation) is separately
// quantified in tb_mmio_top.v and is added as a second column in Table III.
//
// Correctness
// -----------
// For every layer, an integer-arithmetic golden reference is computed inside
// the TB and compared bit-exactly against the fabric output.  Mismatch -> fail.
// This guarantees the dataflow is correct end-to-end; actual MNIST top-1
// accuracy is measured separately on the STM32 baseline with the same weights.
//
// Weights & images
// ----------------
// To keep the TB self-contained and deterministic, weights and images are
// synthesised procedurally (small-magnitude pseudo-random INT16) so that all
// accumulations fit comfortably inside 32-bit signed.  This does NOT train on
// MNIST -- its purpose is to exercise the fabric for representative work.
// =============================================================================
module tb_minicnn;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;
`ifdef REAL_MNIST
    // real MNIST mode: 200 test images, real trained INT4 weights loaded from
    // data/*.mem produced by scripts/train_minicnn.py.  Input pixels are
    // centered (uint8>>2 - 32) so they live in [-32, +31].
    parameter N_IMG  = 200;
`else
    parameter N_IMG  = 3;         // synthetic prand smoke-test mode
`endif

    reg clk = 0; always #5 clk = ~clk;        // 100 MHz sim clock

    // ------------------------------------------------------------------
    //  DUT  (shared-fabric sage16_top, driven directly -- no MMIO bus)
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

    // cycle counter + rising-edge done detector
    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    reg  done_d;
    wire done_rise = done & ~done_d;
    always @(posedge clk) done_d <= done;

    // ------------------------------------------------------------------
    //  Cycle-accumulating task for one fabric call (matmul or conv)
    // ------------------------------------------------------------------
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

    // signed matmul call (mode=3 = MODE_MMS on sage16_top).  Reuses the exact
    // same fabric routing as MODE_MM but toggles the PE opcode to OP_MACB_S
    // so CNN FC layers can multiply signed INT weights by signed activations.
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
    //  Problem storage
    // ------------------------------------------------------------------
    localparam HIN   = 26;         // input  height/width
    localparam HC1   = 24;         // Conv1  output H/W   (HIN - 3 + 1 = 24)
    localparam NC1   = 4;          // Conv1  output channels
    localparam HP    = 6;          // pooled feature H/W  (24 / 4)
    localparam NFLAT = HP*HP*NC1;  // 144
    localparam NFC1  = 32;         // FC1 output dim
    localparam NFC2  = 12;         // FC2 output dim (10 classes padded to 12 = 3 tiles)

    // deterministic pseudo-random helper
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

    // images / weights / activations (signed arithmetic)
    reg signed [DATA_W-1:0] IMG     [0:N_IMG-1][0:HIN-1][0:HIN-1];
    reg signed [DATA_W-1:0] CONV1_K [0:NC1-1][0:2][0:2];
    reg signed [DATA_W-1:0] W1      [0:NFC1-1][0:NFLAT-1];
    reg signed [DATA_W-1:0] W2      [0:NFC2-1][0:NFC1-1];
`ifdef REAL_MNIST
    reg signed [DATA_W-1:0] LABEL   [0:N_IMG-1];   // ground-truth class
`endif

    // per-image intermediate activations
    reg signed [ACC_W-1:0]  A_C1    [0:NC1-1][0:HC1-1][0:HC1-1];
    reg signed [ACC_W-1:0]  A_POOL  [0:NC1-1][0:HP-1][0:HP-1];
    reg signed [ACC_W-1:0]  A_FLAT  [0:NFLAT-1];
    reg signed [ACC_W-1:0]  A_FC1   [0:NFC1-1];
    reg signed [ACC_W-1:0]  A_FC2   [0:NFC2-1];

    // golden (software) references
    reg signed [ACC_W-1:0]  G_C1    [0:NC1-1][0:HC1-1][0:HC1-1];
    reg signed [ACC_W-1:0]  G_POOL  [0:NC1-1][0:HP-1][0:HP-1];
    reg signed [ACC_W-1:0]  G_FC1   [0:NFC1-1];
    reg signed [ACC_W-1:0]  G_FC2   [0:NFC2-1];

    // cycle counters
    integer cy_conv1_tot, cy_pool_tot, cy_fc1_tot, cy_fc2_tot;
    integer cy_conv1_i,   cy_pool_i,   cy_fc1_i,   cy_fc2_i;
    integer cy_sw_tot;
    integer n_pass_c1, n_fail_c1, n_pass_fc1, n_fail_fc1, n_pass_fc2, n_fail_fc2;
    integer n_pass_pred, n_fail_pred;
    // MNIST-mode counters: argmax-vs-ground-truth (only used under REAL_MNIST)
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
    //  Weight / image generation
    // ------------------------------------------------------------------
    task init_weights_and_images;
        integer s;
        begin
`ifdef REAL_MNIST
            // ----- REAL MNIST mode: load trained INT4 weights and centered
            //       int test images dumped by scripts/train_minicnn.py.
            //       Each .mem file is one 4-hex value per line, written in
            //       row-major order matching the verilog packed layout below.
            $readmemh("data/conv1_k.mem", CONV1_K);
            $readmemh("data/w1.mem",      W1);
            $readmemh("data/w2.mem",      W2);
            $readmemh("data/images.mem",  IMG);
            $readmemh("data/labels.mem",  LABEL);
            $display("  [REAL_MNIST] loaded conv1_k(%0d), w1(%0d), w2(%0d), images(%0d), labels(%0d)",
                     NC1*9, NFC1*NFLAT, NFC2*NFC1, N_IMG*HIN*HIN, N_IMG);
`else
            // ----- synthetic mode: small-magnitude prand to exercise the
            //       fabric without needing trained data.
            s = 1;
            for (oc=0; oc<NC1; oc=oc+1)
                for (i=0; i<3; i=i+1)
                    for (j=0; j<3; j=j+1) begin
                        CONV1_K[oc][i][j] = prand(s, -4, 4);
                        s = s + 1;
                    end
            for (i=0; i<NFC1; i=i+1)
                for (j=0; j<NFLAT; j=j+1) begin
                    W1[i][j] = prand(s, -4, 4);
                    s = s + 1;
                end
            for (i=0; i<NFC2; i=i+1)
                for (j=0; j<NFC1; j=j+1) begin
                    if (i >= 10)
                        W2[i][j] = 0;
                    else begin
                        W2[i][j] = prand(s, -4, 4);
                        s = s + 1;
                    end
                end
            for (ch=0; ch<N_IMG; ch=ch+1)
                for (r=0; r<HIN; r=r+1)
                    for (c=0; c<HIN; c=c+1) begin
                        IMG[ch][r][c] = prand(s, 0, 63);
                        s = s + 1;
                    end
`endif
        end
    endtask

    // ------------------------------------------------------------------
    //  Software (golden) reference for one image
    // ------------------------------------------------------------------
    task compute_golden;
        input  integer img_idx;
        integer oc, r, c, i, j;
        reg signed [ACC_W-1:0] acc;
        reg signed [ACC_W-1:0] poolacc;
        integer pi, pj;
        begin
            // Conv1
            for (oc=0; oc<NC1; oc=oc+1)
                for (r=0; r<HC1; r=r+1)
                    for (c=0; c<HC1; c=c+1) begin
                        acc = 0;
                        for (i=0; i<3; i=i+1)
                            for (j=0; j<3; j=j+1)
                                acc = acc + $signed(IMG[img_idx][r+i][c+j])
                                          * $signed(CONV1_K[oc][i][j]);
                        G_C1[oc][r][c] = acc;
                    end
            // AvgPool 4x4 stride 4 (sum, not true avg -- keeps INT exact)
            for (oc=0; oc<NC1; oc=oc+1)
                for (r=0; r<HP; r=r+1)
                    for (c=0; c<HP; c=c+1) begin
                        poolacc = 0;
                        for (pi=0; pi<4; pi=pi+1)
                            for (pj=0; pj<4; pj=pj+1)
                                poolacc = poolacc + G_C1[oc][r*4+pi][c*4+pj];
                        // divide by 16 (arithmetic shift right 4)
                        G_POOL[oc][r][c] = poolacc >>> 4;
                    end
            // FC1 (no activation yet, we keep raw, ReLU after)
            for (oc=0; oc<NFC1; oc=oc+1) begin
                acc = 0;
                for (i=0; i<NFC1+0; i=i+1) ; // dummy
                // flatten POOL into 1D then matmul
                for (j=0; j<NFLAT; j=j+1) begin
                    integer ch_j, r_j, c_j;
                    ch_j = j / (HP*HP);
                    r_j  = (j % (HP*HP)) / HP;
                    c_j  = j % HP;
                    acc = acc + G_POOL[ch_j][r_j][c_j] * $signed(W1[oc][j]);
                end
                G_FC1[oc] = acc;
            end
            // ReLU on FC1 applied in-place for FC2 consumption
            for (i=0; i<NFC1; i=i+1)
                if (G_FC1[i] < 0) G_FC1[i] = 0;
            // FC2 = W2 * ReLU(FC1)
            for (oc=0; oc<NFC2; oc=oc+1) begin
                acc = 0;
                for (j=0; j<NFC1; j=j+1)
                    acc = acc + G_FC1[j] * $signed(W2[oc][j]);
                G_FC2[oc] = acc;
            end
        end
    endtask

    // ------------------------------------------------------------------
    //  Helper: assemble a 6x6 image patch into img_pack, 3x3 kernel into k_pack
    // ------------------------------------------------------------------
    task pack_conv_tile;
        input  integer img_idx;
        input  integer r0, c0;       // top-left of 6x6 input patch (Conv3x3 on 6x6 => 4x4)
        input  integer oc;           // output channel (selects kernel set)
        begin
            for (r=0; r<6; r=r+1)
                for (c=0; c<6; c=c+1)
                    img_pack[(r*6+c)*DATA_W +: DATA_W] =
                        IMG[img_idx][r0+r][c0+c];
            for (i=0; i<3; i=i+1)
                for (j=0; j<3; j=j+1)
                    k_pack[(i*3+j)*DATA_W +: DATA_W] = CONV1_K[oc][i][j];
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 1: Conv1 via 6x6 tiles of conv3x3_sage16
    //   Conv1 output is 24x24.  Tile into 6x6 non-overlapping 4x4 output
    //   blocks => 6 tile-rows * 6 tile-cols * NC1 channels = 144 calls.
    // ------------------------------------------------------------------
    task run_conv1_image;
        input integer img_idx;
        integer tr_out, tc_out;
        integer rr, cc;
        begin
            layer_cycles = 0;
            for (oc=0; oc<NC1; oc=oc+1) begin
                for (tr_out=0; tr_out<HC1; tr_out=tr_out+4) begin
                    for (tc_out=0; tc_out<HC1; tc_out=tc_out+4) begin
                        // pack patch (6x6 input starting at (tr_out, tc_out))
                        for (rr=0; rr<6; rr=rr+1)
                            for (cc=0; cc<6; cc=cc+1)
                                img_pack[(rr*6+cc)*DATA_W +: DATA_W] =
                                    IMG[img_idx][tr_out+rr][tc_out+cc];
                        for (i=0; i<3; i=i+1)
                            for (j=0; j<3; j=j+1)
                                k_pack[(i*3+j)*DATA_W +: DATA_W] =
                                    CONV1_K[oc][i][j];
                        run_conv_call(img_pack, k_pack, c_pack);
                        // unpack into A_C1
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
    //  Layer 2: AvgPool 4x4 stride 4 (SW on host -- represents PicoRV32 work)
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
            // flatten into A_FLAT
            for (oc=0; oc<NC1; oc=oc+1)
                for (r=0; r<HP; r=r+1)
                    for (c=0; c<HP; c=c+1)
                        A_FLAT[oc*(HP*HP) + r*HP + c] = A_POOL[oc][r][c];
            // consume N clock cycles to approximate real SW time:
            // 4 * 6 * 6 * 16 = 2304 adds + 144 shifts ~= 2500 instr on PicoRV32
            // (~1 IPC) => ~2500 cycles.  Emulate by ticking through.
            repeat (2500) @(posedge clk);
            cy_pool_i = cyc_cnt - t0_sw;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 3: FC1 = W1 * A_FLAT  (matmul, 32 x 144  tiled 4x4)
    //
    //  Tile layout: output rows OC=0..NFC1 in groups of 4, inner K dim in
    //  groups of 4.  Each fabric call does C[4x4] += W_tile(4x4) * X_tile(4x4)
    //  where X_tile's column 0 = x[k..k+3] (activation) and cols 1..3 = 0.
    //  Only the first column of the result is meaningful; we accumulate that
    //  into FC1.
    // ------------------------------------------------------------------
    task run_fc1;
        integer oc0, kk, ii, jj;
        reg signed [ACC_W-1:0] sum;
        begin
            layer_cycles = 0;
            for (oc0=0; oc0<NFC1; oc0=oc0+4) begin
                // zero the 4-element partial
                for (ii=0; ii<4; ii=ii+1) tile_acc[ii] = 0;
                for (kk=0; kk<NFLAT; kk=kk+4) begin
                    // A_tile = W1[oc0..oc0+3][kk..kk+3]  (row-major 4x4)
                    for (ii=0; ii<4; ii=ii+1)
                        for (jj=0; jj<4; jj=jj+1)
                            A_tile[(ii*4+jj)*DATA_W +: DATA_W] =
                                W1[oc0+ii][kk+jj];
                    // B_tile: column 0 = A_FLAT[kk..kk+3] (low DATA_W bits),
                    //          columns 1..3 = 0. Row-major layout: B[k][j]
                    //          at index k*4 + j.
                    for (ii=0; ii<4; ii=ii+1) begin
                        B_tile[(ii*4+0)*DATA_W +: DATA_W] =
                            A_FLAT[kk+ii][DATA_W-1:0];
                        B_tile[(ii*4+1)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+2)*DATA_W +: DATA_W] = 0;
                        B_tile[(ii*4+3)*DATA_W +: DATA_W] = 0;
                    end
                    run_matmul_call(A_tile, B_tile, c_pack);
                    // accumulate column-0 results
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

    // ------------------------------------------------------------------
    //  ReLU on A_FC1 (SW)
    // ------------------------------------------------------------------
    task run_relu;
        integer ii;
        begin
            for (ii=0; ii<NFC1; ii=ii+1)
                if (A_FC1[ii] < 0) A_FC1[ii] = 0;
        end
    endtask

    // ------------------------------------------------------------------
    //  Layer 4: FC2 = W2 * ReLU(A_FC1)  (matmul, 12 x 32 tiled)
    //  Same column-broadcast trick as FC1.
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

    task check_fc1_post_relu;
        integer ii;
        reg signed [ACC_W-1:0] want;
        begin
            for (ii=0; ii<NFC1; ii=ii+1) begin
                want = G_FC1[ii];                // already ReLUed in compute_golden
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
            // also tally fabric vs ground-truth MNIST accuracy
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
        $dumpfile("build/minicnn_tb.vcd");
        $dumpvars(0, tb_minicnn);
        rst_n = 0; mode = 0; start = 0;
        mm_a = 0; mm_b = 0; cv_img_r = 0; cv_k_r = 0;
        cy_conv1_tot = 0; cy_pool_tot = 0; cy_fc1_tot = 0; cy_fc2_tot = 0;
        cy_sw_tot = 0;
        n_pass_c1 = 0; n_fail_c1 = 0;
        n_pass_fc1 = 0; n_fail_fc1 = 0;
        n_pass_fc2 = 0; n_fail_fc2 = 0;
        n_pass_pred = 0; n_fail_pred = 0;
        n_correct_label = 0; n_total_label = 0;

        init_weights_and_images;

        #23 rst_n = 1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  MiniCNN-3x3  end-to-end inference on sage16_top");
        $display("  topology: 26x26 -> Conv1(3x3,1->4) -> AvgPool4 ->");
        $display("            FC1(144->32) -> ReLU -> FC2(32->12) -> argmax");
        $display("  %0d test images, per-image per-layer cycle counts", N_IMG);
        $display("================================================");

        // dump up to the first 8 images so the paper figure can render them.
        // Capping the dump keeps the log small in REAL_MNIST mode (200 imgs).
        for (img=0; img<((N_IMG < 8) ? N_IMG : 8); img=img+1) begin
            $display("#IMG_BEGIN %0d  size %0dx%0d", img, HIN, HIN);
            for (r=0; r<HIN; r=r+1) begin
                $write("#IMG_ROW %0d %0d :", img, r);
                for (c=0; c<HIN; c=c+1)
                    $write(" %0d", IMG[img][r][c]);
                $write("\n");
            end
            $display("#IMG_END %0d", img);
        end

        for (img=0; img<N_IMG; img=img+1) begin
            // golden reference (integer sw)
            compute_golden(img);

            // layer 1 -- fabric conv3x3
            run_conv1_image(img);
            cy_conv1_tot = cy_conv1_tot + cy_conv1_i;
            check_conv1;

            // dump Conv1 feature maps for the first image so the paper
            // figure can show "what the fabric actually computed"
            if (img == 0) begin
                for (oc=0; oc<NC1; oc=oc+1) begin
                    $display("#C1_BEGIN %0d size %0dx%0d", oc, HC1, HC1);
                    for (r=0; r<HC1; r=r+1) begin
                        $write("#C1_ROW %0d %0d :", oc, r);
                        for (c=0; c<HC1; c=c+1)
                            $write(" %0d", A_C1[oc][r][c]);
                        $write("\n");
                    end
                    $display("#C1_END %0d", oc);
                end
            end

            // layer 2 -- pool + flatten (sw, emulated)
            run_pool;
            cy_pool_tot  = cy_pool_tot + cy_pool_i;

            // layer 3 -- fabric matmul
            run_fc1;
            cy_fc1_tot   = cy_fc1_tot + cy_fc1_i;

            // ReLU (sw, negligible cycles: bundled into FC2 sw bookkeeping below)
            run_relu;
            check_fc1_post_relu;

            // layer 4 -- fabric matmul
            run_fc2;
            cy_fc2_tot   = cy_fc2_tot + cy_fc2_i;
            check_fc2;

            // argmax
            check_argmax(img);

`ifdef REAL_MNIST
            if (img < 8)
                $display("  img %0d : conv1=%0d  pool=%0d  fc1=%0d  fc2=%0d  pred=%0d (label=%0d)",
                         img, cy_conv1_i, cy_pool_i, cy_fc1_i, cy_fc2_i,
                         argmax_i, LABEL[img]);
`else
            $display("  img %0d : conv1=%0d  pool=%0d  fc1=%0d  fc2=%0d  pred=%0d (exp=%0d)",
                     img, cy_conv1_i, cy_pool_i, cy_fc1_i, cy_fc2_i,
                     argmax_i, argmax_g);
`endif

            // dump the 12-way FC2 logits for the paper figure (cap to 8)
            if (img < 8) begin
                $write("#LOGITS %0d :", img);
                for (i=0; i<NFC2; i=i+1)
                    $write(" %0d", A_FC2[i]);
                $write("\n");
            end
        end

        begin : summary_blk
            integer total_fab;
            integer total_full;
            real    fab_us, full_us;
            total_fab  = cy_conv1_tot + cy_fc1_tot + cy_fc2_tot;
            total_full = total_fab + cy_pool_tot;
            fab_us     = (total_fab  / (1.0*N_IMG)) / 160.0;  // us at 160 MHz
            full_us    = (total_full / (1.0*N_IMG)) / 160.0;
            $display("");
            $display("================================================");
            $display("  MiniCNN  SUMMARY   (totals over %0d images)", N_IMG);
            $display("================================================");
            $display("  fabric compute cycles    : Conv1=%0d  FC1=%0d  FC2=%0d",
                     cy_conv1_tot, cy_fc1_tot, cy_fc2_tot);
            $display("  host SW cycles (pool)    : %0d", cy_pool_tot);
            $display("  per-image avg            : fabric=%0d  full=%0d",
                     total_fab/N_IMG, total_full/N_IMG);
            $display("  at 160 MHz               : fabric=%0.1f us  full=%0.1f us",
                     fab_us, full_us);
            $display("  bit-exact check          : conv1 pass=%0d fail=%0d",
                     n_pass_c1, n_fail_c1);
            $display("                            fc1   pass=%0d fail=%0d",
                     n_pass_fc1, n_fail_fc1);
            $display("                            fc2   pass=%0d fail=%0d",
                     n_pass_fc2, n_fail_fc2);
            $display("  pred matches golden      : %0d / %0d",
                     n_pass_pred, N_IMG);
`ifdef REAL_MNIST
            $display("  MNIST top-1 (fabric vs ground truth)");
            $display("                            : %0d / %0d  (%0.2f%% accuracy)",
                     n_correct_label, n_total_label,
                     100.0 * n_correct_label / (1.0 * n_total_label));
`endif
            $display("  SUMMARY: pass=%0d fail=%0d",
                     n_pass_c1 + n_pass_fc1 + n_pass_fc2 + n_pass_pred,
                     n_fail_c1 + n_fail_fc1 + n_fail_fc2 + n_fail_pred);
            $display("================================================");
        end

        #100 $finish;
    end
endmodule
