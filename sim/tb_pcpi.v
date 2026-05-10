`timescale 1ns/1ps
// =============================================================================
// tb_pcpi.v
//
// Drives the sage16_pcpi module exactly the way PicoRV32 would: assert
// pcpi_valid + pcpi_insn + pcpi_rs1/rs2, then wait for pcpi_ready.
// Demonstrates a full 4x4 matmul sequence using ONLY the four custom
// instructions. This is the software-level interface the RISC-V core will use.
// =============================================================================
module tb_pcpi;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0;
    always #5 clk = ~clk;          // 100 MHz

    reg         rst_n;
    reg         pcpi_valid;
    reg  [31:0] pcpi_insn;
    reg  [31:0] pcpi_rs1;
    reg  [31:0] pcpi_rs2;
    wire        pcpi_wr;
    wire [31:0] pcpi_rd;
    wire        pcpi_wait;
    wire        pcpi_ready;

    sage16_pcpi #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut(
        .clk(clk), .rst_n(rst_n),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready)
    );

    // ---- encode a custom-0 R-type instruction ----
    function [31:0] enc_r;
        input [6:0] funct7;
        input [4:0] rs2;
        input [4:0] rs1;
        input [2:0] funct3;
        input [4:0] rd;
        begin
            enc_r = {funct7, rs2, rs1, funct3, rd, 7'b0001011};
        end
    endfunction

    localparam [2:0] F_WRA = 3'd0, F_WRB = 3'd1, F_GO = 3'd2, F_RDC = 3'd3;

    // ---- PCPI "issue one instruction" (like what picorv32 does) ----
    task pcpi_issue;
        input [31:0] insn;
        input [31:0] rs1v;
        input [31:0] rs2v;
        output [31:0] rd_val;
        begin
            @(negedge clk);
            pcpi_valid = 1;
            pcpi_insn  = insn;
            pcpi_rs1   = rs1v;
            pcpi_rs2   = rs2v;
            rd_val     = 32'd0;
            // hold until pcpi_ready observed
            wait(pcpi_ready);
            if(pcpi_wr) rd_val = pcpi_rd;
            @(negedge clk);
            pcpi_valid = 0;
            pcpi_insn  = 0;
            pcpi_rs1   = 0;
            pcpi_rs2   = 0;
        end
    endtask

    task sage16_wrA; input [3:0] idx; input [DATA_W-1:0] val;
        reg [31:0] dummy;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_WRA,5'd0),{28'd0,idx},{16'd0,val},dummy); end
    endtask
    task sage16_wrB; input [3:0] idx; input [DATA_W-1:0] val;
        reg [31:0] dummy;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_WRB,5'd0),{28'd0,idx},{16'd0,val},dummy); end
    endtask
    task sage16_go;
        reg [31:0] dummy;
        begin pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_GO,5'd0),32'd0,32'd0,dummy); end
    endtask
    task sage16_rdC; input [3:0] idx; output [ACC_W-1:0] val;
        reg [31:0] rd_val;
        begin
            pcpi_issue(enc_r(7'd0,5'd0,5'd0,F_RDC,5'd0),{28'd0,idx},32'd0,rd_val);
            val = rd_val[ACC_W-1:0];
        end
    endtask

    // ---- a self-checking matmul over PCPI ----
    reg  [DATA_W-1:0] A [0:3][0:3], B [0:3][0:3];
    reg  [ACC_W-1:0]  Cref [0:3][0:3];
    integer i,j,k,idx;
    integer pass, fail;
    reg  [ACC_W-1:0]  got;

    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    integer t_go_begin, t_go_end;

    initial begin
        $dumpfile("build/pcpi_tb.vcd"); $dumpvars(0, tb_pcpi);
        rst_n=0; pcpi_valid=0; pcpi_insn=0; pcpi_rs1=0; pcpi_rs2=0;
        #23 rst_n=1;
        @(posedge clk); #1;

        // Fill A, B
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            A[i][j] = (i*4+j+1);          // 1..16
            B[i][j] = (i==j) ? 2 : 1;     // diag=2, off=1
        end
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            Cref[i][j] = 0;
            for(k=0;k<4;k=k+1) Cref[i][j] = Cref[i][j] + A[i][k]*B[k][j];
        end

        // stream A[]/B[] via sage16.wrA / sage16.wrB
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            idx = i*4+j;
            sage16_wrA(idx[3:0], A[i][j]);
            sage16_wrB(idx[3:0], B[i][j]);
        end

        // fire matmul
        t_go_begin = cyc_cnt;
        sage16_go;
        t_go_end = cyc_cnt;

        // read back C[]
        pass = 0; fail = 0;
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            idx = i*4+j;
            sage16_rdC(idx[3:0], got);
            if(got === Cref[i][j]) pass = pass + 1;
            else begin
                fail = fail + 1;
                $display("  MISMATCH C[%0d][%0d]: got=%0d exp=%0d", i,j,got,Cref[i][j]);
            end
        end

        $display("");
        $display("================================================");
        $display("  PCPI matmul:   pass=%0d fail=%0d", pass, fail);
        $display("  sage16.go latency (cycles, incl. PCPI handshake): %0d",
                 t_go_end - t_go_begin);
        $display("================================================");

        #50 $finish;
    end
endmodule
