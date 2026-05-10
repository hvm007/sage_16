`timescale 1ns/1ps
// =============================================================================
// tb_pcpi_top.v - drives sage16_pcpi_top exactly the way PicoRV32 would.
//
// Runs matmul, conv3x3 and quaternion multiply sequentially through the
// 4-instruction custom-0 API: sage16.setMode / sage16.wrOp / sage16.go / sage16.rdC.
// =============================================================================
module tb_pcpi_top;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0; always #5 clk = ~clk;       // 100 MHz

    reg         rst_n;
    reg         pcpi_valid;
    reg  [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2;
    wire        pcpi_wr;
    wire [31:0] pcpi_rd;
    wire        pcpi_wait;
    wire        pcpi_ready;

    sage16_pcpi_top #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut(
        .clk(clk), .rst_n(rst_n),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready)
    );

    // ---- R-type encoder ----
    function [31:0] enc_r;
        input [6:0] funct7; input [4:0] rs2; input [4:0] rs1;
        input [2:0] funct3; input [4:0] rd;
        begin enc_r = {funct7, rs2, rs1, funct3, rd, 7'b0001011}; end
    endfunction

    localparam [2:0] F_WROP = 3'd0, F_GO = 3'd2, F_RDC = 3'd3, F_MODE = 3'd4;

    // ---- one PCPI transaction ----
    task pcpi_issue;
        input  [31:0] insn;
        input  [31:0] rs1v, rs2v;
        output [31:0] rd_val;
        begin
            @(negedge clk);
            pcpi_valid = 1;
            pcpi_insn  = insn;
            pcpi_rs1   = rs1v;
            pcpi_rs2   = rs2v;
            rd_val     = 32'd0;
            wait(pcpi_ready);
            if(pcpi_wr) rd_val = pcpi_rd;
            @(negedge clk);
            pcpi_valid = 0; pcpi_insn = 0; pcpi_rs1 = 0; pcpi_rs2 = 0;
        end
    endtask

    task sage16_setMode; input [1:0] m;
        reg [31:0] d;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_MODE,5'd0),{30'd0,m},32'd0,d); end
    endtask
    task sage16_wrOp; input [5:0] idx; input [DATA_W-1:0] val;
        reg [31:0] d;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_WROP,5'd0),{26'd0,idx},{16'd0,val},d); end
    endtask
    task sage16_go;
        reg [31:0] d;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_GO,5'd0),32'd0,32'd0,d); end
    endtask
    task sage16_rdC; input [3:0] idx; output [ACC_W-1:0] val;
        reg [31:0] d;
        begin
            pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_RDC,5'd0),{28'd0,idx},32'd0,d);
            val = d[ACC_W-1:0];
        end
    endtask

    // ---- cycle counter ----
    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    // ---- reference tables ----
    reg [DATA_W-1:0]        A [0:3][0:3], B [0:3][0:3];
    reg [ACC_W-1:0]         Cmm_ref [0:15];
    reg [DATA_W-1:0]        IMG [0:5][0:5];
    reg signed [DATA_W-1:0] K [0:2][0:2];
    reg signed [ACC_W-1:0]  Ccv_ref [0:15];
    reg signed [DATA_W-1:0] Q2 [0:3][0:3];
    reg signed [ACC_W-1:0]  Cqt_ref [0:15];

    integer i, jj, k_, r, c, di, dj, ir, ic;
    integer pass_mm, fail_mm, pass_cv, fail_cv, pass_qt, fail_qt;
    integer lat_mm, lat_cv, lat_qt, t0;
    reg [5:0] op_idx_tmp;
    reg [3:0] rd_idx_tmp;
    reg [ACC_W-1:0] got_w;
    reg signed [ACC_W-1:0] got_s;
    reg signed [ACC_W-1:0] acc_tmp, pix;
    reg signed [ACC_W-1:0] w1,x1,y1,z1,w2,x2,y2,z2;

    initial begin
        $dumpfile("build/pcpi_top_tb.vcd");
        $dumpvars(0, tb_pcpi_top);
        rst_n=0; pcpi_valid=0; pcpi_insn=0; pcpi_rs1=0; pcpi_rs2=0;
        pass_mm=0; fail_mm=0; pass_cv=0; fail_cv=0; pass_qt=0; fail_qt=0;
        #23 rst_n=1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  sage16_pcpi_top  :  matmul + conv3x3 + quat via PCPI");
        $display("================================================");

        // ========== matmul ==========
        sage16_setMode(2'd0);
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            A[i][jj] = i*4 + jj + 1;
            B[i][jj] = (i==jj) ? 2 : 1;
        end
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            Cmm_ref[i*4+jj] = 0;
            for(k_=0;k_<4;k_=k_+1)
                Cmm_ref[i*4+jj] = Cmm_ref[i*4+jj] + A[i][k_]*B[k_][jj];
        end
        for(i=0;i<16;i=i+1) begin
            ir = i/4; ic = i%4;
            op_idx_tmp = i[5:0];
            sage16_wrOp(op_idx_tmp, A[ir][ic]);
        end
        for(i=0;i<16;i=i+1) begin
            ir = i/4; ic = i%4;
            op_idx_tmp = (i+16);
            sage16_wrOp(op_idx_tmp, B[ir][ic]);
        end
        t0 = cyc_cnt; sage16_go; lat_mm = cyc_cnt - t0;
        for(i=0;i<16;i=i+1) begin
            rd_idx_tmp = i[3:0];
            sage16_rdC(rd_idx_tmp, got_w);
            if(got_w === Cmm_ref[i][ACC_W-1:0]) pass_mm = pass_mm + 1;
            else begin
                fail_mm = fail_mm + 1;
                $display("  [matmul] idx=%0d got=%0d exp=%0d",
                         i, got_w, Cmm_ref[i]);
            end
        end
        $display("  [matmul ] pass=%0d/16  sage16.go latency=%0d cy",
                 pass_mm, lat_mm);

        // ========== conv3x3 (Sobel-X on a ramp) ==========
        sage16_setMode(2'd1);
        for(r=0;r<6;r=r+1) for(c=0;c<6;c=c+1) IMG[r][c] = 10*r + c;
        K[0][0]=-1; K[0][1]=0; K[0][2]=1;
        K[1][0]=-2; K[1][1]=0; K[1][2]=2;
        K[2][0]=-1; K[2][1]=0; K[2][2]=1;
        for(r=0;r<4;r=r+1) for(c=0;c<4;c=c+1) begin
            acc_tmp = 0;
            for(di=0;di<3;di=di+1) for(dj=0;dj<3;dj=dj+1) begin
                pix = $signed({16'd0, IMG[r+di][c+dj]});
                acc_tmp = acc_tmp + $signed(K[di][dj]) * pix;
            end
            Ccv_ref[r*4+c] = acc_tmp;
        end
        for(r=0;r<6;r=r+1) for(c=0;c<6;c=c+1) begin
            op_idx_tmp = r*6+c;
            sage16_wrOp(op_idx_tmp, IMG[r][c]);
        end
        for(di=0;di<3;di=di+1) for(dj=0;dj<3;dj=dj+1) begin
            op_idx_tmp = 6'd36 + di*3 + dj;
            sage16_wrOp(op_idx_tmp, K[di][dj]);
        end
        t0 = cyc_cnt; sage16_go; lat_cv = cyc_cnt - t0;
        for(i=0;i<16;i=i+1) begin
            rd_idx_tmp = i[3:0];
            sage16_rdC(rd_idx_tmp, got_w);
            got_s = $signed(got_w);
            if(got_s === Ccv_ref[i]) pass_cv = pass_cv + 1;
            else begin
                fail_cv = fail_cv + 1;
                $display("  [conv] idx=%0d got=%0d exp=%0d",
                         i, got_s, Ccv_ref[i]);
            end
        end
        $display("  [conv3x3] pass=%0d/16  sage16.go latency=%0d cy",
                 pass_cv, lat_cv);

        // ========== quat (q1=i) ==========
        sage16_setMode(2'd2);
        sage16_wrOp(6'd0, 16'sd0);  // Q1.w
        sage16_wrOp(6'd1, 16'sd1);  // Q1.x
        sage16_wrOp(6'd2, 16'sd0);  // Q1.y
        sage16_wrOp(6'd3, 16'sd0);  // Q1.z
        Q2[0][0]=0; Q2[0][1]=0; Q2[0][2]=1; Q2[0][3]=0;
        Q2[1][0]=0; Q2[1][1]=0; Q2[1][2]=0; Q2[1][3]=1;
        Q2[2][0]=0; Q2[2][1]=1; Q2[2][2]=0; Q2[2][3]=0;
        Q2[3][0]=1; Q2[3][1]=0; Q2[3][2]=0; Q2[3][3]=0;
        for(jj=0;jj<4;jj=jj+1) for(k_=0;k_<4;k_=k_+1) begin
            op_idx_tmp = 6'd4 + jj*4 + k_;
            sage16_wrOp(op_idx_tmp, Q2[jj][k_]);
        end
        w1=0; x1=1; y1=0; z1=0;
        for(jj=0;jj<4;jj=jj+1) begin
            w2=$signed(Q2[jj][0]); x2=$signed(Q2[jj][1]);
            y2=$signed(Q2[jj][2]); z2=$signed(Q2[jj][3]);
            Cqt_ref[jj*4+0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
            Cqt_ref[jj*4+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
            Cqt_ref[jj*4+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
            Cqt_ref[jj*4+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
        end
        t0 = cyc_cnt; sage16_go; lat_qt = cyc_cnt - t0;
        for(i=0;i<16;i=i+1) begin
            rd_idx_tmp = i[3:0];
            sage16_rdC(rd_idx_tmp, got_w);
            got_s = $signed(got_w);
            if(got_s === Cqt_ref[i]) pass_qt = pass_qt + 1;
            else begin
                fail_qt = fail_qt + 1;
                $display("  [quat] idx=%0d got=%0d exp=%0d",
                         i, got_s, Cqt_ref[i]);
            end
        end
        $display("  [quat   ] pass=%0d/16  sage16.go latency=%0d cy",
                 pass_qt, lat_qt);

        $display("");
        $display("================================================");
        $display("  sage16_pcpi_top SUMMARY: pass=%0d fail=%0d / 48",
                 pass_mm+pass_cv+pass_qt, fail_mm+fail_cv+fail_qt);
        $display("================================================");
        #50 $finish;
    end
endmodule
