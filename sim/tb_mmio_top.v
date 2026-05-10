`timescale 1ns/1ps
// =============================================================================
// tb_mmio_top.v - exercises sage16_mmio_top over the 32-bit register bus.
//
// Runs three kernels back-to-back through pure bus transactions:
//   1. matmul   (unsigned, 4x4)
//   2. conv3x3  (signed, Sobel-X)
//   3. quat     (signed, Hamilton basis i*j=k)
//
// Every operand and every result is delivered via bus_write/bus_read, so
// this TB also measures the realistic end-to-end bus-level latency that
// matters for the paper's "how expensive is software control" story.
// =============================================================================
module tb_mmio_top;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;
    parameter ADDR_W = 10;

    reg clk = 0; always #5 clk = ~clk;              // 100 MHz

    reg                   rst_n;
    reg  [ADDR_W-1:0]     addr;
    reg  [31:0]           wdata;
    wire [31:0]           rdata;
    reg                   we, re;
    wire                  ack, irq;

    sage16_mmio_top #(.DATA_W(DATA_W), .ACC_W(ACC_W), .ADDR_W(ADDR_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .addr(addr), .wdata(wdata), .rdata(rdata),
        .we(we), .re(re), .ack(ack), .irq(irq)
    );

    // ---- bus primitives ----
    task bus_write; input [ADDR_W-1:0] a; input [31:0] d;
        begin
            @(negedge clk);
            addr = a; wdata = d; we = 1; re = 0;
            @(posedge clk); #1;
            we = 0; addr = 0; wdata = 0;
        end
    endtask

    task bus_read; input [ADDR_W-1:0] a; output [31:0] d;
        begin
            @(negedge clk);
            addr = a; re = 1; we = 0;
            @(posedge clk); #1;
            d  = rdata;
            re = 0; addr = 0;
        end
    endtask

    // ---- register map constants ----
    localparam [ADDR_W-1:0] REG_OPBUF_BASE = 10'h000;   // + 4*idx
    localparam [ADDR_W-1:0] REG_RESBUF_BASE = 10'h100;  // + 4*idx
    localparam [ADDR_W-1:0] REG_MODE       = 10'h200;
    localparam [ADDR_W-1:0] REG_CTRL       = 10'h204;
    localparam [ADDR_W-1:0] REG_STATUS     = 10'h208;

    // ---- cycle counter for latency measurement ----
    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    // ---- per-kernel driver tasks ----
    integer i;
    reg [31:0] status;

    task set_mode_and_go;
        input [1:0]  m;
        output integer cyc_gogetdone;
        integer t0;
        begin
            bus_write(REG_MODE, {30'd0, m});
            t0 = cyc_cnt;
            bus_write(REG_CTRL, 32'h1);
            // poll STATUS until bit0 (done) is high
            status = 0;
            while (status[0] == 1'b0) begin
                bus_read(REG_STATUS, status);
            end
            cyc_gogetdone = cyc_cnt - t0;
        end
    endtask

    task write_op;  input [5:0] idx; input [DATA_W-1:0] val;
        begin
            bus_write(REG_OPBUF_BASE + {4'd0, idx, 2'd0}, {16'd0, val});
        end
    endtask

    task read_res;  input [3:0] idx; output [31:0] val;
        begin
            bus_read(REG_RESBUF_BASE + {4'd0, idx, 2'd0}, val);
        end
    endtask

    // ---- reference tables ----
    reg [DATA_W-1:0]        A [0:3][0:3], B [0:3][0:3];
    reg [ACC_W-1:0]         Cmm_ref [0:15];
    reg [DATA_W-1:0]        IMG [0:5][0:5];
    reg signed [DATA_W-1:0] K   [0:2][0:2];
    reg signed [ACC_W-1:0]  Ccv_ref [0:15];
    reg signed [DATA_W-1:0] Q2  [0:3][0:3];
    reg signed [ACC_W-1:0]  Cqt_ref [0:15];

    integer pass_mm, fail_mm, pass_cv, fail_cv, pass_qt, fail_qt;
    integer lat_mm, lat_cv, lat_qt;
    reg [31:0] got_w;
    reg signed [ACC_W-1:0] got_s;
    integer r, c, k_, di, dj, jj;
    reg signed [ACC_W-1:0] acc_tmp, pix;
    reg signed [ACC_W-1:0] w1,x1,y1,z1,w2,x2,y2,z2;

    initial begin
        $dumpfile("build/mmio_top_tb.vcd");
        $dumpvars(0, tb_mmio_top);
        rst_n=0; addr=0; wdata=0; we=0; re=0;
        pass_mm=0; fail_mm=0; pass_cv=0; fail_cv=0; pass_qt=0; fail_qt=0;
        #23 rst_n=1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  sage16_mmio_top  :  matmul + conv3x3 + quat via MMIO");
        $display("================================================");

        // ========================================================
        //  PHASE 1 : matmul
        // ========================================================
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            A[i][jj] = i*4 + jj + 1;
            B[i][jj] = (i==jj) ? 2 : 1;
        end
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            Cmm_ref[i*4+jj] = 0;
            for(k_=0;k_<4;k_=k_+1)
                Cmm_ref[i*4+jj] = Cmm_ref[i*4+jj] + A[i][k_]*B[k_][jj];
        end
        // write A into opbuf[0..15], B into opbuf[16..31]
        for(i=0;i<4;i=i+1) for(jj=0;jj<4;jj=jj+1) begin
            write_op(i*4+jj,      A[i][jj]);
            write_op(i*4+jj + 16, B[i][jj]);
        end
        set_mode_and_go(2'd0, lat_mm);

        for(i=0;i<16;i=i+1) begin
            read_res(i[3:0], got_w);
            if(got_w === Cmm_ref[i][ACC_W-1:0]) pass_mm = pass_mm + 1;
            else begin
                fail_mm = fail_mm + 1;
                $display("  [matmul] idx=%0d got=%0d exp=%0d",
                         i, got_w, Cmm_ref[i]);
            end
        end
        $display("  [matmul ]  pass=%0d/16  bus-latency(go->done)=%0d cy",
                 pass_mm, lat_mm);

        // ========================================================
        //  PHASE 2 : conv3x3  (Sobel-X on a ramp image)
        // ========================================================
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
        // write image into opbuf[0..35], kernel into opbuf[36..44]
        for(r=0;r<6;r=r+1) for(c=0;c<6;c=c+1)
            write_op(r*6+c, IMG[r][c]);
        for(di=0;di<3;di=di+1) for(dj=0;dj<3;dj=dj+1)
            write_op(6'd36 + di*3 + dj, K[di][dj]);
        set_mode_and_go(2'd1, lat_cv);

        for(i=0;i<16;i=i+1) begin
            read_res(i[3:0], got_w);
            got_s = $signed(got_w);
            if(got_s === Ccv_ref[i]) pass_cv = pass_cv + 1;
            else begin
                fail_cv = fail_cv + 1;
                $display("  [conv] idx=%0d got=%0d exp=%0d",
                         i, got_s, Ccv_ref[i]);
            end
        end
        $display("  [conv3x3]  pass=%0d/16  bus-latency(go->done)=%0d cy",
                 pass_cv, lat_cv);

        // ========================================================
        //  PHASE 3 : quat (Hamilton basis q1=i)
        // ========================================================
        // Q1 = i (0,1,0,0), and four Q2 vectors
        //   Q2[0]=j, Q2[1]=k, Q2[2]=i, Q2[3]=1
        // so expected results: i*j=k, i*k=-j, i*i=-1, i*1=i
        write_op(0, 16'sd0);  // Q1.w
        write_op(1, 16'sd1);  // Q1.x
        write_op(2, 16'sd0);  // Q1.y
        write_op(3, 16'sd0);  // Q1.z
        Q2[0][0]=0; Q2[0][1]=0; Q2[0][2]=1; Q2[0][3]=0;   // j
        Q2[1][0]=0; Q2[1][1]=0; Q2[1][2]=0; Q2[1][3]=1;   // k
        Q2[2][0]=0; Q2[2][1]=1; Q2[2][2]=0; Q2[2][3]=0;   // i
        Q2[3][0]=1; Q2[3][1]=0; Q2[3][2]=0; Q2[3][3]=0;   // 1
        for(jj=0;jj<4;jj=jj+1) for(k_=0;k_<4;k_=k_+1)
            write_op(6'd4 + jj*4 + k_, Q2[jj][k_]);
        w1=0; x1=1; y1=0; z1=0;
        for(jj=0;jj<4;jj=jj+1) begin
            w2=$signed(Q2[jj][0]); x2=$signed(Q2[jj][1]);
            y2=$signed(Q2[jj][2]); z2=$signed(Q2[jj][3]);
            Cqt_ref[jj*4+0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
            Cqt_ref[jj*4+1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
            Cqt_ref[jj*4+2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
            Cqt_ref[jj*4+3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
        end
        set_mode_and_go(2'd2, lat_qt);

        for(i=0;i<16;i=i+1) begin
            read_res(i[3:0], got_w);
            got_s = $signed(got_w);
            if(got_s === Cqt_ref[i]) pass_qt = pass_qt + 1;
            else begin
                fail_qt = fail_qt + 1;
                $display("  [quat] idx=%0d got=%0d exp=%0d",
                         i, got_s, Cqt_ref[i]);
            end
        end
        $display("  [quat   ]  pass=%0d/16  bus-latency(go->done)=%0d cy",
                 pass_qt, lat_qt);

        $display("");
        $display("================================================");
        $display("  sage16_mmio_top SUMMARY: pass=%0d fail=%0d / 48",
                 pass_mm+pass_cv+pass_qt, fail_mm+fail_cv+fail_qt);
        $display("     matmul   : bus-latency %0d cy", lat_mm);
        $display("     conv3x3  : bus-latency %0d cy", lat_cv);
        $display("     quat     : bus-latency %0d cy", lat_qt);
        $display("     (bus latency includes ctrl-write + status-poll loop)");
        $display("================================================");
        #50 $finish;
    end
endmodule
