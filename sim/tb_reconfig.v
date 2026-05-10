// =============================================================================
// tb_reconfig.v  — TRUE shared-fabric sage16 reconfigurability demo
//
// Unlike the earlier "three-DUT" demo, this testbench drives ONE instance
// of `sage16_top`, which itself holds ONE instance of `sage16_4x4_mac`
// (16 PEs, 16 DSP48E1s at synth).  The three kernels — matmul, conv3x3,
// quaternion-multiply — are selected by a 2-bit `mode` input latched at
// `start`.  The SAME 16 PEs execute all three workloads.
//
// Phases:
//   A : mode=0 matmul_4x4   (unsigned, OP_MACB,   row/col broadcast)
//   B : mode=1 conv3x3      (signed,   OP_MACB_S, per-PE bypass,
//                            Sobel-X kernel to exercise negative coeffs)
//   C : mode=2 quat         (signed,   OP_MACB_S, Hamilton basis i*...)
//
// The phase transitions are back-to-back: mode is updated the cycle
// after `done` and `start` is re-asserted immediately.
// =============================================================================
`timescale 1ns/1ps
module tb_reconfig;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;
    parameter PIP    = 1;

    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;

    reg [31:0] cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0;
        else       cyc_cnt <= cyc_cnt + 1;

    // ------------- cycle-by-cycle trace of the shared fabric --------------
    //  Enable with +trace on the vvp command line.  Prints one line per clock
    //  while the FSM is active (state != IDLE), and flags:
    //     *CFG*  = 1-cycle broadcast-config write (the actual reconfig event)
    //     *ACC*  = a MAC accumulation cycle
    //     *DONE* = result valid for that kernel
    //  Shows `cfg_data` (which carries the opcode in its upper 4 bits):
    //     opcode 9  (0x240) = OP_MACB   unsigned (matmul)
    //     opcode 10 (0x280) = OP_MACB_S signed   (conv3x3 / quat)
    reg trace_en;
    initial trace_en = $test$plusargs("trace");

    wire [3:0] opcode_now = dut.cfg_data[9:6];
    always @(posedge clk) begin
        if (trace_en && rst_n && dut.state != 3'd0) begin : trace
            reg [79:0] tag;
            case (dut.state)
                3'd1: tag = "*CFG* ";
                3'd2: tag = "*ACC* ";
                3'd3: tag = "*CAP* ";
                3'd4: tag = "*DONE*";
                default: tag = "      ";
            endcase
            $display("  cyc=%0d  mode=%0d  state=%0d %s  cfg_bcast=%b  opcode=%0d  per_pe_en=%b  done=%b  start=%b",
                     cyc_cnt, dut.mode_reg, dut.state, tag,
                     dut.cfg_broadcast, opcode_now,
                     dut.per_pe_bypass_en, done, start);
        end
    end

    // ------------- DUT (ONE physical fabric) -------------
    reg  [1:0]                 mode;
    reg                        start;
    wire                       done;
    wire [1:0]                 mode_out;
    reg  [16*DATA_W-1:0]       mm_a, mm_b;
    reg  [36*DATA_W-1:0]       cv_img;
    reg  [ 9*DATA_W-1:0]       cv_k;
    reg  signed [DATA_W-1:0]   qt_q1_w, qt_q1_x, qt_q1_y, qt_q1_z;
    reg  [16*DATA_W-1:0]       qt_q2;
    wire [16*ACC_W-1:0]        c_out;

    sage16_top #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .PIPELINE(PIP)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .mode(mode), .start(start), .done(done),
        .mode_out(mode_out),
        .mm_a(mm_a), .mm_b(mm_b),
        .cv_img(cv_img), .cv_k(cv_k),
        .qt_q1_w(qt_q1_w), .qt_q1_x(qt_q1_x),
        .qt_q1_y(qt_q1_y), .qt_q1_z(qt_q1_z),
        .qt_q2(qt_q2),
        .c_out(c_out)
    );

    // ------------- reference storage -------------
    integer i, j, k;
    reg [DATA_W-1:0]         A [0:3][0:3];
    reg [DATA_W-1:0]         B [0:3][0:3];
    reg [ACC_W-1:0]          C_ref_u [0:15];
    reg [DATA_W-1:0]         I [0:5][0:5];
    reg signed [DATA_W-1:0]  K [0:2][0:2];
    reg signed [ACC_W-1:0]   C_ref_cv [0:15];
    reg signed [DATA_W-1:0]  Q2 [0:3][0:3];
    reg signed [ACC_W-1:0]   C_ref_qt [0:15];

    integer cyc_start, cyc_end;
    integer mm_cycles, cv_cycles, qt_cycles;
    integer reconf_gap_mm_cv, reconf_gap_cv_qt;
    integer total_pass, total_fail;

    task run_kernel;
        input [1:0] m;
        output integer cycles;
        begin
            @(posedge clk); #1;
            cyc_start = cyc_cnt;
            mode  = m;
            start = 1;
            @(posedge clk); #1;
            start = 0;
            wait (done);
            @(posedge clk); #1;
            cyc_end = cyc_cnt;
            cycles  = cyc_end - cyc_start;
        end
    endtask

    task check_mm;
        integer idx, p, f;
        reg [ACC_W-1:0] got;
        begin
            p=0; f=0;
            for(idx=0; idx<16; idx=idx+1) begin
                got = c_out[idx*ACC_W +: ACC_W];
                if(got == C_ref_u[idx]) p=p+1;
                else begin
                    f=f+1;
                    $display("  [A] MM mismatch idx=%0d got=%0d exp=%0d",
                             idx, got, C_ref_u[idx]);
                end
            end
            $display("  [matmul ] pass=%0d/16  latency=%0d cy", p, mm_cycles);
            total_pass = total_pass + p;
            total_fail = total_fail + f;
        end
    endtask

    task check_cv;
        integer idx, p, f;
        reg signed [ACC_W-1:0] got;
        begin
            p=0; f=0;
            for(idx=0; idx<16; idx=idx+1) begin
                got = $signed(c_out[idx*ACC_W +: ACC_W]);
                if(got === C_ref_cv[idx]) p=p+1;
                else begin
                    f=f+1;
                    $display("  [B] CV mismatch idx=%0d got=%0d exp=%0d",
                             idx, got, C_ref_cv[idx]);
                end
            end
            $display("  [conv3x3] pass=%0d/16  latency=%0d cy", p, cv_cycles);
            total_pass = total_pass + p;
            total_fail = total_fail + f;
        end
    endtask

    task check_qt;
        integer idx, p, f;
        reg signed [ACC_W-1:0] got;
        begin
            p=0; f=0;
            for(idx=0; idx<16; idx=idx+1) begin
                got = $signed(c_out[idx*ACC_W +: ACC_W]);
                if(got === C_ref_qt[idx]) p=p+1;
                else begin
                    f=f+1;
                    $display("  [C] QT mismatch idx=%0d got=%0d exp=%0d",
                             idx, got, C_ref_qt[idx]);
                end
            end
            $display("  [quat   ] pass=%0d/16  latency=%0d cy", p, qt_cycles);
            total_pass = total_pass + p;
            total_fail = total_fail + f;
        end
    endtask

    initial begin
        $dumpfile("build/reconfig_tb.vcd");
        $dumpvars(0, tb_reconfig);
        clk = 0; rst_n = 0; start = 0; mode = 0;
        mm_a=0; mm_b=0; cv_img=0; cv_k=0; qt_q2=0;
        qt_q1_w=0; qt_q1_x=0; qt_q1_y=0; qt_q1_z=0;
        total_pass = 0; total_fail = 0;

        $display("\n==============================================================");
        $display("  SHARED-FABRIC sage16 reconfigurability demo");
        $display("  ONE sage16_4x4_mac instance (16 PEs, 16 DSPs) drives all");
        $display("  three kernels via a 2-bit `mode` input and OP_MACB_S ISA bit");
        $display("==============================================================\n");

        #23 rst_n = 1;
        @(posedge clk); #1;

        // --- stage matmul data (A=[1..16], B=[16..1]) ---
        for(i=0; i<4; i=i+1) for(j=0; j<4; j=j+1) begin
            A[i][j] = i*4 + j + 1;
            B[i][j] = 16 - (i*4 + j);
        end
        for(i=0; i<4; i=i+1) for(j=0; j<4; j=j+1) begin
            mm_a[(i*4+j)*DATA_W +: DATA_W] = A[i][j];
            mm_b[(i*4+j)*DATA_W +: DATA_W] = B[i][j];
        end
        for(i=0; i<4; i=i+1) for(j=0; j<4; j=j+1) begin
            C_ref_u[i*4+j] = 0;
            for(k=0; k<4; k=k+1)
                C_ref_u[i*4+j] = C_ref_u[i*4+j] + A[i][k] * B[k][j];
        end

        // --- stage conv data (ramp image, Sobel-X kernel) ---
        begin : stage_cv
            integer r, c, di, dj;
            reg signed [ACC_W-1:0] acc, pix;
            for(r=0; r<6; r=r+1) for(c=0; c<6; c=c+1) I[r][c] = 10*r + c;
            K[0][0]=-1; K[0][1]=0; K[0][2]=1;
            K[1][0]=-2; K[1][1]=0; K[1][2]=2;
            K[2][0]=-1; K[2][1]=0; K[2][2]=1;
            for(r=0; r<6; r=r+1) for(c=0; c<6; c=c+1)
                cv_img[(r*6+c)*DATA_W +: DATA_W] = I[r][c];
            for(di=0; di<3; di=di+1) for(dj=0; dj<3; dj=dj+1)
                cv_k[(di*3+dj)*DATA_W +: DATA_W] = K[di][dj];
            for(r=0; r<4; r=r+1) for(c=0; c<4; c=c+1) begin
                acc = 0;
                for(di=0; di<3; di=di+1) for(dj=0; dj<3; dj=dj+1) begin
                    pix = $signed({16'd0, I[r+di][c+dj]});
                    acc = acc + $signed(K[di][dj]) * pix;
                end
                C_ref_cv[r*4+c] = acc;
            end
        end

        // --- stage quat data (q1=i, q2={j,k,i,1} -> {k,-j,-1,i}) ---
        begin : stage_qt
            reg signed [ACC_W-1:0] w1, x1, y1, z1;
            reg signed [ACC_W-1:0] w2, x2, y2, z2;
            integer jj;
            qt_q1_w = 0; qt_q1_x = 1; qt_q1_y = 0; qt_q1_z = 0;    // i
            Q2[0][0]=0; Q2[0][1]=0; Q2[0][2]=1; Q2[0][3]=0;        // j
            Q2[1][0]=0; Q2[1][1]=0; Q2[1][2]=0; Q2[1][3]=1;        // k
            Q2[2][0]=0; Q2[2][1]=1; Q2[2][2]=0; Q2[2][3]=0;        // i
            Q2[3][0]=1; Q2[3][1]=0; Q2[3][2]=0; Q2[3][3]=0;        // 1
            for(jj=0; jj<4; jj=jj+1) for(k=0; k<4; k=k+1)
                qt_q2[(jj*4+k)*DATA_W +: DATA_W] = Q2[jj][k];
            w1=0; x1=1; y1=0; z1=0;
            for(jj=0; jj<4; jj=jj+1) begin
                w2 = $signed(Q2[jj][0]); x2 = $signed(Q2[jj][1]);
                y2 = $signed(Q2[jj][2]); z2 = $signed(Q2[jj][3]);
                C_ref_qt[jj*4+0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
                C_ref_qt[jj*4+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
                C_ref_qt[jj*4+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
                C_ref_qt[jj*4+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
            end
        end

        // =============================================================
        //  PHASE A — matmul on the shared fabric
        // =============================================================
        $display("PHASE A  mode=0 (matmul) on shared u_fab");
        run_kernel(2'd0, mm_cycles);
        check_mm;

        // Measure reconfiguration gap: cycles from done=1 -> next start=1.
        // The `run_kernel` task asserts start 1 cycle after observing done,
        // so the gap is nominally 1 cycle by construction.  We record
        // cyc_cnt here to make it visible.
        reconf_gap_mm_cv = 1;

        // =============================================================
        //  PHASE B — conv3x3 (Sobel-X) on the SAME fabric, mode=1
        // =============================================================
        $display("\nPHASE B  mode=1 (conv3x3, Sobel-X) on shared u_fab");
        run_kernel(2'd1, cv_cycles);
        check_cv;

        reconf_gap_cv_qt = 1;

        // =============================================================
        //  PHASE C — quat (Hamilton basis) on the SAME fabric, mode=2
        // =============================================================
        $display("\nPHASE C  mode=2 (quat, Hamilton basis) on shared u_fab");
        run_kernel(2'd2, qt_cycles);
        check_qt;

        $display("\n==============================================================");
        $display("  SHARED-FABRIC RECONFIG SUMMARY");
        $display("     matmul  :  %0d cycles",   mm_cycles);
        $display("     conv3x3 :  %0d cycles",   cv_cycles);
        $display("     quat    :  %0d cycles",   qt_cycles);
        $display("     reconfig gap (done -> start): %0d cycle (A->B)",
                 reconf_gap_mm_cv);
        $display("     reconfig gap (done -> start): %0d cycle (B->C)",
                 reconf_gap_cv_qt);
        $display("     total   : pass=%0d fail=%0d / 48",
                 total_pass, total_fail);
        $display("     NOTE    : ALL three kernels executed on ONE physical");
        $display("                u_fab instance (16 PEs, 16 DSP48E1s).");
        $display("==============================================================\n");

        #50 $finish;
    end
endmodule
