`timescale 1ns/1ps
// Exhaustive directed mode-transition test for the shared SAGE-16 fabric.
// Every ordered pair among matmul, conv3x3, and quaternion is executed
// back-to-back with no reset between the two kernels.
module tb_reconfig_pairs;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n, start;
    reg [1:0] mode;
    wire done;
    wire [1:0] mode_out;
    reg  [16*DATA_W-1:0] mm_a, mm_b;
    reg  [36*DATA_W-1:0] cv_img;
    reg  [ 9*DATA_W-1:0] cv_k;
    reg  signed [DATA_W-1:0] qt_q1_w, qt_q1_x, qt_q1_y, qt_q1_z;
    reg  [16*DATA_W-1:0] qt_q2;
    wire [16*ACC_W-1:0] c_out;

    sage16_top #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n), .mode(mode), .start(start),
        .done(done), .mode_out(mode_out),
        .mm_a(mm_a), .mm_b(mm_b), .cv_img(cv_img), .cv_k(cv_k),
        .qt_q1_w(qt_q1_w), .qt_q1_x(qt_q1_x),
        .qt_q1_y(qt_q1_y), .qt_q1_z(qt_q1_z), .qt_q2(qt_q2),
        .c_out(c_out)
    );

    reg [DATA_W-1:0] A [0:3][0:3], B [0:3][0:3];
    reg [ACC_W-1:0] Cmm [0:15];
    reg [DATA_W-1:0] IMG [0:5][0:5];
    reg signed [DATA_W-1:0] K [0:2][0:2];
    reg signed [ACC_W-1:0] Ccv [0:15];
    reg signed [DATA_W-1:0] Q2 [0:3][0:3];
    reg signed [ACC_W-1:0] Cqt [0:15];

    integer i, j, k, r, c, di, dj;
    integer pass, fail, transitions, cfg_cycles;
    reg signed [ACC_W-1:0] acc, pix, got_s;
    reg signed [ACC_W-1:0] w1, x1, y1, z1, w2, x2, y2, z2;

    task check_output;
        input [1:0] active_mode;
        integer idx;
        begin
            for (idx=0; idx<16; idx=idx+1) begin
                got_s = $signed(c_out[idx*ACC_W +: ACC_W]);
                case (active_mode)
                    2'd0: if (c_out[idx*ACC_W +: ACC_W] === Cmm[idx])
                              pass = pass + 1;
                           else begin
                              fail = fail + 1;
                              $display("  MM mismatch idx=%0d got=%0d exp=%0d",
                                       idx, c_out[idx*ACC_W +: ACC_W], Cmm[idx]);
                           end
                    2'd1: if (got_s === Ccv[idx])
                              pass = pass + 1;
                           else begin
                              fail = fail + 1;
                              $display("  CV mismatch idx=%0d got=%0d exp=%0d",
                                       idx, got_s, Ccv[idx]);
                           end
                    2'd2: if (got_s === Cqt[idx])
                              pass = pass + 1;
                           else begin
                              fail = fail + 1;
                              $display("  QT mismatch idx=%0d got=%0d exp=%0d",
                                       idx, got_s, Cqt[idx]);
                           end
                    default: fail = fail + 1;
                endcase
            end
        end
    endtask

    task run_pair;
        input [1:0] first_mode;
        input [1:0] second_mode;
        integer cfg_before;
        begin
            rst_n = 0;
            start = 0;
            mode = first_mode;
            repeat (3) @(posedge clk);
            #1 rst_n = 1;
            @(negedge clk);
            start = 1;

            // Pre-stage the second mode after the first mode is safely latched.
            wait (dut.state == 3'd2 && dut.mode_reg == first_mode);
            @(negedge clk);
            mode = second_mode;

            // Capture and check the first result in its S_DONE cycle.
            wait (dut.state == 3'd3 && dut.mode_reg == first_mode);
            @(posedge clk); #1;
            check_output(first_mode);

            // Exactly one configuration-broadcast cycle must separate kernels.
            cfg_before = cfg_cycles;
            wait (dut.state == 3'd2 && dut.mode_reg == second_mode);
            if ((cfg_cycles - cfg_before) != 1) begin
                fail = fail + 1;
                $display("  CFG length mismatch %0d->%0d: got %0d cycles",
                         first_mode, second_mode, cfg_cycles - cfg_before);
            end
            transitions = transitions + 1;
            @(negedge clk);
            start = 0;

            wait (dut.state == 3'd3 && dut.mode_reg == second_mode);
            @(posedge clk); #1;
            check_output(second_mode);
            wait (dut.state == 3'd0);
            $display("  pair %0d->%0d complete; cumulative pass=%0d fail=%0d",
                     first_mode, second_mode, pass, fail);
        end
    endtask

    always @(posedge clk)
        if (rst_n && dut.cfg_broadcast)
            cfg_cycles = cfg_cycles + 1;

    initial begin
        mm_a=0; mm_b=0; cv_img=0; cv_k=0; qt_q2=0;
        qt_q1_w=0; qt_q1_x=0; qt_q1_y=0; qt_q1_z=0;
        rst_n=0; start=0; mode=0;
        pass=0; fail=0; transitions=0; cfg_cycles=0;

        // Matmul: signed-safe positive operands.
        for (i=0;i<4;i=i+1) for (j=0;j<4;j=j+1) begin
            A[i][j] = i*4+j+1;
            B[i][j] = (i==j) ? 2 : 1;
            mm_a[(i*4+j)*DATA_W +: DATA_W] = A[i][j];
            mm_b[(i*4+j)*DATA_W +: DATA_W] = B[i][j];
        end
        for (i=0;i<4;i=i+1) for (j=0;j<4;j=j+1) begin
            Cmm[i*4+j] = 0;
            for (k=0;k<4;k=k+1)
                Cmm[i*4+j] = Cmm[i*4+j] + A[i][k]*B[k][j];
        end

        // Convolution: Sobel-X on a 6x6 ramp.
        for (r=0;r<6;r=r+1) for (c=0;c<6;c=c+1) begin
            IMG[r][c] = 10*r+c;
            cv_img[(r*6+c)*DATA_W +: DATA_W] = IMG[r][c];
        end
        K[0][0]=-1; K[0][1]=0; K[0][2]=1;
        K[1][0]=-2; K[1][1]=0; K[1][2]=2;
        K[2][0]=-1; K[2][1]=0; K[2][2]=1;
        for (di=0;di<3;di=di+1) for (dj=0;dj<3;dj=dj+1)
            cv_k[(di*3+dj)*DATA_W +: DATA_W] = K[di][dj];
        for (r=0;r<4;r=r+1) for (c=0;c<4;c=c+1) begin
            acc = 0;
            for (di=0;di<3;di=di+1) for (dj=0;dj<3;dj=dj+1) begin
                pix = $signed({16'd0, IMG[r+di][c+dj]});
                acc = acc + $signed(K[di][dj]) * pix;
            end
            Ccv[r*4+c] = acc;
        end

        // Quaternion: q1=i, q2={j,k,i,1}.
        qt_q1_w=0; qt_q1_x=1; qt_q1_y=0; qt_q1_z=0;
        Q2[0][0]=0; Q2[0][1]=0; Q2[0][2]=1; Q2[0][3]=0;
        Q2[1][0]=0; Q2[1][1]=0; Q2[1][2]=0; Q2[1][3]=1;
        Q2[2][0]=0; Q2[2][1]=1; Q2[2][2]=0; Q2[2][3]=0;
        Q2[3][0]=1; Q2[3][1]=0; Q2[3][2]=0; Q2[3][3]=0;
        for (j=0;j<4;j=j+1) for (k=0;k<4;k=k+1)
            qt_q2[(j*4+k)*DATA_W +: DATA_W] = Q2[j][k];
        w1=0; x1=1; y1=0; z1=0;
        for (j=0;j<4;j=j+1) begin
            w2=Q2[j][0]; x2=Q2[j][1]; y2=Q2[j][2]; z2=Q2[j][3];
            Cqt[j*4+0] = w1*w2-x1*x2-y1*y2-z1*z2;
            Cqt[j*4+1] = w1*x2+x1*w2+y1*z2-z1*y2;
            Cqt[j*4+2] = w1*y2-x1*z2+y1*w2+z1*x2;
            Cqt[j*4+3] = w1*z2+x1*y2-y1*x2+z1*w2;
        end

        run_pair(2'd0, 2'd1);
        run_pair(2'd0, 2'd2);
        run_pair(2'd1, 2'd0);
        run_pair(2'd1, 2'd2);
        run_pair(2'd2, 2'd0);
        run_pair(2'd2, 2'd1);

        $display("");
        $display("================================================");
        $display("  DIRECTED RECONFIG SUMMARY");
        $display("  transitions: %0d/6, cfg event: one cycle each", transitions);
        $display("  outputs: pass=%0d fail=%0d / 192", pass, fail);
        $display("================================================");
        #50 $finish;
    end
endmodule
