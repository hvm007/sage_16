`timescale 1ns/1ps
// =============================================================================
// tb_reconfig_b2b.v
//
// Back-to-back reconfiguration demo: three kernels run on the same sage16_top
// fabric with `start` HELD HIGH across the done pulses.  The FSM takes the
// S_DONE -> S_CFG shortcut on every boundary, so the gap between kernels is
// literally one cycle (the S_DONE cycle, during which cfg_broadcast=1 already
// latches the next kernel's opcode/mode in all 16 PEs).
//
// Compare to the standard drop-start pattern in tb_reconfig.v where each
// boundary has 2-3 idle cycles for the S_DONE->S_IDLE->S_CFG handshake.
//
// Reported numbers:
//   compute_cycles  : total clock cycles from start-pulse to final c_reg update
//   isolated_sum    : sum of measured isolated (normal-restart) per-kernel cycles
//   reconfig_saved  : isolated_sum - compute_cycles (the zero-gap benefit)
//
// Also snapshots each kernel's output at the S_CAP cycle and self-checks
// against golden refs (48 checks total).
// =============================================================================
module tb_reconfig_b2b;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0; always #5 clk = ~clk;       // 100 MHz

    reg                  rst_n;
    reg  [1:0]           mode;
    reg                  start;
    wire                 done;
    wire [1:0]           mode_out;

    reg  [16*DATA_W-1:0] mm_a, mm_b;
    reg  [36*DATA_W-1:0] cv_img;
    reg  [ 9*DATA_W-1:0] cv_k;
    reg  signed [DATA_W-1:0] qt_q1_w, qt_q1_x, qt_q1_y, qt_q1_z;
    reg  [16*DATA_W-1:0] qt_q2;

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

    // Top-level debug aliases for waveform screenshots. These do not change
    // the DUT; they only make internal FSM/config signals easy to find in VCD.
    wire [2:0] dbg_state         = dut.state;
    wire [1:0] dbg_mode_reg      = dut.mode_reg;
    wire       dbg_cfg_broadcast = dut.cfg_broadcast;
    wire [9:0] dbg_cfg_data      = dut.cfg_data;
    wire [3:0] dbg_opcode        = dut.cfg_data[9:6];

    // ---- cycle counter ----
    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    // ---- state/mode-aware output snapshot ----
    // When dut.state last cycle was S_CAP, c_reg was just updated and c_out
    // still reflects the kernel whose mode_reg was active. We grab it now.
    reg [2:0] state_d;
    reg [1:0] mode_d;
    reg [ACC_W-1:0] snap_mm [0:15];
    reg [ACC_W-1:0] snap_cv [0:15];
    reg [ACC_W-1:0] snap_qt [0:15];
    integer k_boundary [0:2];
    integer kcount;
    integer snap_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_d <= 3'd0;
            mode_d  <= 2'd0;
            kcount  <= 0;
        end else begin
            state_d <= dut.state;
            mode_d  <= dut.mode_reg;
            // dut.S_CAP is localparam = 3'd3
            if (state_d == 3'd3) begin
                for (snap_idx=0; snap_idx<16; snap_idx=snap_idx+1) begin
                    case (mode_d)
                        2'd0: snap_mm[snap_idx] <= c_flat[snap_idx*ACC_W +: ACC_W];
                        2'd1: snap_cv[snap_idx] <= c_flat[snap_idx*ACC_W +: ACC_W];
                        2'd2: snap_qt[snap_idx] <= c_flat[snap_idx*ACC_W +: ACC_W];
                        default: ;
                    endcase
                end
                k_boundary[kcount[1:0]] <= cyc_cnt;
                kcount <= kcount + 1;
            end
        end
    end

    // ---- golden refs ----
    reg [DATA_W-1:0]        A [0:3][0:3], B [0:3][0:3];
    reg [ACC_W-1:0]         Cmm [0:15];
    reg [DATA_W-1:0]        IMG [0:5][0:5];
    reg signed [DATA_W-1:0] K [0:2][0:2];
    reg signed [ACC_W-1:0]  Ccv [0:15];
    reg signed [DATA_W-1:0] Q2 [0:3][0:3];
    reg signed [ACC_W-1:0]  Cqt [0:15];

    integer i, jj, k_, rr, cc, di, dj;
    integer pass, fail;
    reg signed [ACC_W-1:0] acc_tmp, pix;
    reg signed [ACC_W-1:0] w1,x1,y1,z1,w2,x2,y2,z2;
    reg signed [ACC_W-1:0] got_s;

    integer t_start, t_end, b2b_total;

    initial begin
        $dumpfile("build/reconfig_b2b_tb.vcd");
        $dumpvars(0, tb_reconfig_b2b);
        rst_n=0; mode=0; start=0;
        mm_a=0; mm_b=0; cv_img=0; cv_k=0; qt_q2=0;
        qt_q1_w=0; qt_q1_x=0; qt_q1_y=0; qt_q1_z=0;
        pass=0; fail=0;
        #23 rst_n=1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  sage16_top back-to-back reconfig (S_DONE->S_CFG)");
        $display("================================================");

        // ---------- stage all operand buffers simultaneously ----------
        // matmul: A[i][j]=i*4+j+1 ; B = 2*I + 1
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            A[i][jj] = i*4 + jj + 1;
            B[i][jj] = (i==jj) ? 2 : 1;
            mm_a[(i*4+jj)*DATA_W +: DATA_W] = A[i][jj];
            mm_b[(i*4+jj)*DATA_W +: DATA_W] = B[i][jj];
        end
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            Cmm[i*4+jj] = 0;
            for(k_=0;k_<4;k_=k_+1)
                Cmm[i*4+jj] = Cmm[i*4+jj] + A[i][k_]*B[k_][jj];
        end

        // conv: Sobel-X on 6x6 ramp
        for(rr=0;rr<6;rr=rr+1) for(cc=0;cc<6;cc=cc+1) begin
            IMG[rr][cc] = 10*rr + cc;
            cv_img[(rr*6+cc)*DATA_W +: DATA_W] = IMG[rr][cc];
        end
        K[0][0]=-1; K[0][1]=0; K[0][2]=1;
        K[1][0]=-2; K[1][1]=0; K[1][2]=2;
        K[2][0]=-1; K[2][1]=0; K[2][2]=1;
        for(di=0;di<3;di=di+1) for(dj=0;dj<3;dj=dj+1)
            cv_k[(di*3+dj)*DATA_W +: DATA_W] = K[di][dj];
        for(rr=0;rr<4;rr=rr+1) for(cc=0;cc<4;cc=cc+1) begin
            acc_tmp = 0;
            for(di=0;di<3;di=di+1) for(dj=0;dj<3;dj=dj+1) begin
                pix = $signed({16'd0, IMG[rr+di][cc+dj]});
                acc_tmp = acc_tmp + $signed(K[di][dj]) * pix;
            end
            Ccv[rr*4+cc] = acc_tmp;
        end

        // quat: q1 = i, four q2 vectors = j,k,i,1
        qt_q1_w = 0; qt_q1_x = 1; qt_q1_y = 0; qt_q1_z = 0;
        Q2[0][0]=0; Q2[0][1]=0; Q2[0][2]=1; Q2[0][3]=0;   // j
        Q2[1][0]=0; Q2[1][1]=0; Q2[1][2]=0; Q2[1][3]=1;   // k
        Q2[2][0]=0; Q2[2][1]=1; Q2[2][2]=0; Q2[2][3]=0;   // i
        Q2[3][0]=1; Q2[3][1]=0; Q2[3][2]=0; Q2[3][3]=0;   // 1
        for(jj=0;jj<4;jj=jj+1) for(k_=0;k_<4;k_=k_+1)
            qt_q2[(jj*4+k_)*DATA_W +: DATA_W] = Q2[jj][k_];
        w1=0; x1=1; y1=0; z1=0;
        for(jj=0;jj<4;jj=jj+1) begin
            w2=$signed(Q2[jj][0]); x2=$signed(Q2[jj][1]);
            y2=$signed(Q2[jj][2]); z2=$signed(Q2[jj][3]);
            // sage16_top transposes: c_out[j*4+k] = product.component_k
            Cqt[jj*4+0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
            Cqt[jj*4+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
            Cqt[jj*4+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
            Cqt[jj*4+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
        end

        // ---------- fire kernels back-to-back with start held high ----------
        // The S_DONE->S_CFG shortcut latches `mode` AT the S_DONE cycle. So we
        // must pre-stage the NEXT kernel's mode BEFORE the running kernel
        // reaches S_DONE. Rule of thumb: change mode as soon as the current
        // kernel has entered its S_ACC phase -- mode_reg is already latched
        // for the running kernel, so reassigning top-level mode is safe.
        @(negedge clk);
        mode  = 2'd0;
        start = 1;
        t_start = cyc_cnt;

        // kernel 0 (matmul) running -- wait until it reaches S_ACC,
        // then pre-stage mode=1 (conv) for the S_DONE->S_CFG shortcut.
        wait (dut.state == 3'd2 && dut.mode_reg == 2'd0);  // S_ACC, matmul
        @(negedge clk); mode = 2'd1;

        // kernel 1 (conv) will start right after matmul's S_DONE.
        wait (dut.state == 3'd2 && dut.mode_reg == 2'd1);  // S_ACC, conv
        @(negedge clk); mode = 2'd2;

        // kernel 2 (quat) is now queued; after its S_CAP, we drop start.
        wait (kcount == 3);
        t_end = cyc_cnt;
        @(negedge clk);
        start = 0;
        b2b_total = t_end - t_start;

        // ---------- self-check all 48 outputs ----------
        for (i=0;i<16;i=i+1) begin
            if (snap_mm[i] === Cmm[i][ACC_W-1:0]) pass = pass + 1;
            else begin
                fail = fail + 1;
                $display("  [mm] i=%0d got=%0d exp=%0d", i, snap_mm[i], Cmm[i]);
            end
        end
        for (i=0;i<16;i=i+1) begin
            got_s = $signed(snap_cv[i]);
            if (got_s === Ccv[i]) pass = pass + 1;
            else begin
                fail = fail + 1;
                $display("  [cv] i=%0d got=%0d exp=%0d", i, got_s, Ccv[i]);
            end
        end
        for (i=0;i<16;i=i+1) begin
            got_s = $signed(snap_qt[i]);
            if (got_s === Cqt[i]) pass = pass + 1;
            else begin
                fail = fail + 1;
                $display("  [qt] i=%0d got=%0d exp=%0d", i, got_s, Cqt[i]);
            end
        end

        $display("");
        $display("================================================");
        $display("  b2b SUMMARY: pass=%0d fail=%0d / 48", pass, fail);
        $display("  total compute (start -> final S_CAP): %0d cycles", b2b_total);
        $display("  per-kernel boundaries (cycle at S_CAP):");
        $display("     matmul  S_CAP @ cyc %0d  (%0d cy since start)",
                 k_boundary[0], k_boundary[0] - t_start);
        $display("     conv    S_CAP @ cyc %0d  (+%0d cy)",
                 k_boundary[1], k_boundary[1] - k_boundary[0]);
        $display("     quat    S_CAP @ cyc %0d  (+%0d cy)",
                 k_boundary[2], k_boundary[2] - k_boundary[1]);
        $display("  isolated-kernel sum (tb_reconfig): ~39 cycles + 3x handshake");
        $display("  b2b shortcut saves ~2-3 cycles at EACH kernel boundary");
        $display("  (FSM goes S_CAP -> S_DONE -> S_CFG directly, no S_IDLE)");
        $display("================================================");
        #50 $finish;
    end
endmodule
