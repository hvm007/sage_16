`timescale 1ns/1ps
// =============================================================================
// tb_matmul_tiled.v
//
// Runs 8x8 and 16x16 matmul by 4x4 tiling on the shared sage16_top fabric,
// with the tile controller lifted entirely into software (SW-level loops
// inside the TB). This is exactly the code the RISC-V core would execute
// if the fabric was memory-mapped -- so the measured cycle count is a
// realistic upper bound for the MMIO-driven scaling story.
//
// For each output tile C[I,J]:
//     C[I,J] = sum over K of A[I,K] * B[K,J]
// We issue (N/4) kernel calls per output tile, read the 4x4 result each
// time, and accumulate in a local tile buffer.  The tile is then copied
// into the global result.  No state is carried across invocations -- the
// fabric is stateless-between-kernels, which is exactly the paper's
// scaling claim.
//
// Reports:
//   - compute_cycles : sum of kernel busy cycles only (fabric active)
//   - end2end_cycles : wall-clock from first start to last done
//   - mac_efficiency : MACs-done / (cycles * 16 PEs)
// =============================================================================
module tb_matmul_tiled;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0; always #5 clk = ~clk;       // 100 MHz

    reg                  rst_n;
    reg  [1:0]           mode;
    reg                  start;
    wire                 done;
    wire [1:0]           mode_out;

    reg  [16*DATA_W-1:0] mm_a, mm_b;
    wire [36*DATA_W-1:0] cv_img = 0;
    wire [ 9*DATA_W-1:0] cv_k   = 0;
    wire [DATA_W-1:0]    qt_q1_w = 0, qt_q1_x = 0, qt_q1_y = 0, qt_q1_z = 0;
    wire [16*DATA_W-1:0] qt_q2   = 0;
    wire [16*ACC_W-1:0]  c_flat;

    sage16_top #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .mode(mode), .start(start), .done(done), .mode_out(mode_out),
        .mm_a(mm_a), .mm_b(mm_b),
        .cv_img(cv_img), .cv_k(cv_k),
        .qt_q1_w(qt_q1_w), .qt_q1_x(qt_q1_x),
        .qt_q1_y(qt_q1_y), .qt_q1_z(qt_q1_z),
        .qt_q2(qt_q2),
        .c_out(c_flat)
    );

    // ---- cycle counter ----
    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    // rising-edge detect on done, so multi-kernel sequences wait for the
    // NEW done, not the leftover one from the previous tile
    reg  done_d;
    wire done_rise = done & ~done_d;
    always @(posedge clk) done_d <= done;

    // ---- one 4x4 matmul invocation; returns C = A*B ----
    task run_tile_4x4;
        input  [16*DATA_W-1:0] A4, B4;
        output [16*ACC_W-1:0]  C4;
        input  integer         first_tile;   // 1 iff this is the very first call
        integer t0;
        begin
            @(negedge clk);
            mm_a  = A4;
            mm_b  = B4;
            mode  = 2'd0;
            start = 1;
            @(posedge clk); #1;
            start = 0;
            t0 = cyc_cnt;
            // Wait for the NEW done pulse. On first_tile=1 the previous
            // done was clear anyway; on later tiles we ignore the leftover
            // high until a rising edge re-fires.
            wait(done_rise);
            @(posedge clk); #1;
            compute_cycles = compute_cycles + (cyc_cnt - t0);
            C4 = c_flat;
            if (first_tile) ;  // just silence linter
        end
    endtask

    // ---- problem data (up to 16x16) ----
    localparam NMAX = 16;
    reg  [DATA_W-1:0] Abig [0:NMAX-1][0:NMAX-1];
    reg  [DATA_W-1:0] Bbig [0:NMAX-1][0:NMAX-1];
    reg  [ACC_W-1:0]  Cgot [0:NMAX-1][0:NMAX-1];
    reg  [ACC_W-1:0]  Cref [0:NMAX-1][0:NMAX-1];

    // ---- tile buffers ----
    reg  [16*DATA_W-1:0] A_tile, B_tile;
    reg  [16*ACC_W-1:0]  C_tile;
    reg  [ACC_W-1:0]     C_tile_acc [0:15];

    integer compute_cycles;
    integer t_start, t_end;
    integer N, I, J, K, rr, cc, kk, pass, fail;
    integer first;

    // ---- pack a 4x4 slice of Abig at (row0,col0) into a flat 256b bus ----
    task pack_tile;
        input  integer srcsel;  // 0=Abig, 1=Bbig
        input  integer row0;
        input  integer col0;
        output [16*DATA_W-1:0] tile;
        reg  [DATA_W-1:0] v;
        begin
            for (rr=0; rr<4; rr=rr+1) begin
                for (cc=0; cc<4; cc=cc+1) begin
                    if (srcsel == 0) v = Abig[row0+rr][col0+cc];
                    else             v = Bbig[row0+rr][col0+cc];
                    tile[(rr*4+cc)*DATA_W +: DATA_W] = v;
                end
            end
        end
    endtask

    // ---- full NxN matmul via 4x4 tiling ----
    task run_tiled;
        input  integer NN;
        output integer compute_out;
        output integer e2e_out;
        begin
            // clear output and tile accumulators
            for (rr=0;rr<NN;rr=rr+1) for (cc=0;cc<NN;cc=cc+1) Cgot[rr][cc] = 0;
            compute_cycles = 0;
            t_start = cyc_cnt;
            first = 1;
            for (I=0; I<NN; I=I+4) begin
                for (J=0; J<NN; J=J+4) begin
                    for (kk=0; kk<16; kk=kk+1) C_tile_acc[kk] = 0;
                    for (K=0; K<NN; K=K+4) begin
                        pack_tile(0, I, K, A_tile);
                        pack_tile(1, K, J, B_tile);
                        run_tile_4x4(A_tile, B_tile, C_tile, first);
                        first = 0;
                        for (kk=0; kk<16; kk=kk+1)
                            C_tile_acc[kk] = C_tile_acc[kk]
                                           + C_tile[kk*ACC_W +: ACC_W];
                    end
                    // write tile back to Cgot
                    for (rr=0; rr<4; rr=rr+1) for (cc=0; cc<4; cc=cc+1)
                        Cgot[I+rr][J+cc] = C_tile_acc[rr*4+cc];
                end
            end
            t_end = cyc_cnt;
            compute_out = compute_cycles;
            e2e_out     = t_end - t_start;
        end
    endtask

    integer ccmp, ce2e;
    integer total_pass, total_fail;
    real    eff;

    initial begin
        $dumpfile("build/tiled_tb.vcd");
        $dumpvars(0, tb_matmul_tiled);
        rst_n = 0; mode = 0; start = 0; mm_a = 0; mm_b = 0;
        total_pass = 0; total_fail = 0;
        #23 rst_n = 1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  Tiled matmul scaling study on sage16_top");
        $display("================================================");

        // =========================
        //  N = 8
        // =========================
        N = 8;
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            Abig[rr][cc] = rr + cc + 1;
            Bbig[rr][cc] = (rr == cc) ? 2 : 1;
        end
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            Cref[rr][cc] = 0;
            for (kk=0; kk<N; kk=kk+1)
                Cref[rr][cc] = Cref[rr][cc] + Abig[rr][kk]*Bbig[kk][cc];
        end
        run_tiled(N, ccmp, ce2e);
        pass = 0; fail = 0;
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            if (Cgot[rr][cc] === Cref[rr][cc]) pass = pass + 1;
            else begin
                fail = fail + 1;
                if (fail <= 4) $display("  [8x8] C[%0d][%0d]: got=%0d exp=%0d",
                                        rr, cc, Cgot[rr][cc], Cref[rr][cc]);
            end
        end
        // efficiency = MACs / (cycles * 16 PEs).  MACs = N^3
        eff = (1.0 * N * N * N) / (ccmp * 16.0);
        $display("  [ 8x 8]  pass=%0d/%0d  compute=%0d cy  e2e=%0d cy  MAC_eff=%0.3f",
                 pass, N*N, ccmp, ce2e, eff);
        total_pass = total_pass + pass; total_fail = total_fail + fail;

        // =========================
        //  N = 16
        // =========================
        N = 16;
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            Abig[rr][cc] = rr + cc + 1;
            Bbig[rr][cc] = (rr == cc) ? 2 : 1;
        end
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            Cref[rr][cc] = 0;
            for (kk=0; kk<N; kk=kk+1)
                Cref[rr][cc] = Cref[rr][cc] + Abig[rr][kk]*Bbig[kk][cc];
        end
        run_tiled(N, ccmp, ce2e);
        pass = 0; fail = 0;
        for (rr=0; rr<N; rr=rr+1) for (cc=0; cc<N; cc=cc+1) begin
            if (Cgot[rr][cc] === Cref[rr][cc]) pass = pass + 1;
            else begin
                fail = fail + 1;
                if (fail <= 4) $display("  [16x16] C[%0d][%0d]: got=%0d exp=%0d",
                                        rr, cc, Cgot[rr][cc], Cref[rr][cc]);
            end
        end
        eff = (1.0 * N * N * N) / (ccmp * 16.0);
        $display("  [16x16]  pass=%0d/%0d  compute=%0d cy  e2e=%0d cy  MAC_eff=%0.3f",
                 pass, N*N, ccmp, ce2e, eff);
        total_pass = total_pass + pass; total_fail = total_fail + fail;

        $display("");
        $display("================================================");
        $display("  tiled matmul SUMMARY: pass=%0d fail=%0d / %0d",
                 total_pass, total_fail, 8*8 + 16*16);
        $display("  scaling: 4x4 -> 8x8 -> 16x16 compute cycles");
        $display("  (MAC_eff peaks at 1.0 when all 16 PEs do useful work each cy)");
        $display("================================================");
        #50 $finish;
    end
endmodule
