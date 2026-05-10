`timescale 1ns/1ps
// Exercises the MMIO interface exactly the way AXI-Lite / a CPU would.
module tb_mmio;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;

    reg clk = 0; always #5 clk = ~clk;  // 100 MHz

    reg         rst_n;
    reg  [7:0]  addr;
    reg  [31:0] wdata;
    wire [31:0] rdata;
    reg         we, re;
    wire        ack, irq;

    sage16_mmio #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut(
        .clk(clk), .rst_n(rst_n),
        .addr(addr), .wdata(wdata), .rdata(rdata),
        .we(we), .re(re), .ack(ack), .irq(irq)
    );

    // ---- bus primitives ----
    task bus_write; input [7:0] a; input [31:0] d;
        begin
            @(negedge clk);
            addr = a; wdata = d; we = 1; re = 0;
            @(posedge clk); #1;
            we = 0; addr = 0; wdata = 0;
        end
    endtask

    task bus_read; input [7:0] a; output [31:0] d;
        begin
            @(negedge clk);
            addr = a; re = 1; we = 0;
            @(posedge clk); #1;
            d  = rdata;
            re = 0; addr = 0;
        end
    endtask

    // ---- self-checking matmul ----
    reg [DATA_W-1:0] A[0:3][0:3], B[0:3][0:3];
    reg [ACC_W-1:0]  Cref[0:3][0:3];
    integer i, j, k, idx, pass, fail;
    reg [31:0] r;
    reg [31:0] status;

    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;
    integer t0, t1;

    initial begin
        $dumpfile("build/mmio_tb.vcd"); $dumpvars(0, tb_mmio);
        rst_n = 0; addr=0; wdata=0; we=0; re=0;
        #23 rst_n = 1;
        @(posedge clk); #1;

        // full 16-bit random values (stresses the 32-bit accumulator)
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            A[i][j] = $random;
            B[i][j] = $random;
        end
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            Cref[i][j] = 0;
            for(k=0;k<4;k=k+1) Cref[i][j] = Cref[i][j] + A[i][k]*B[k][j];
        end

        // --- write A buffer (addrs 0x00..0x3C) ---
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            idx = i*4 + j;
            bus_write(8'(idx*4), {16'd0, A[i][j]});
        end
        // --- write B buffer (addrs 0x40..0x7C) ---
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            idx = i*4 + j;
            bus_write(8'(8'h40 + idx*4), {16'd0, B[i][j]});
        end

        // --- kick off (CTRL=0xC0 <= 1) ---
        t0 = cyc_cnt;
        bus_write(8'hC0, 32'd1);

        // --- poll STATUS until done ---
        do begin bus_read(8'hC4, status); end
        while (status[0] == 1'b0);
        t1 = cyc_cnt;

        // --- read C buffer ---
        pass = 0; fail = 0;
        for(i=0;i<4;i=i+1) for(j=0;j<4;j=j+1) begin
            idx = i*4 + j;
            bus_read(8'(8'h80 + idx*4), r);
            if(r === Cref[i][j][ACC_W-1:0]) pass = pass + 1;
            else begin
                fail = fail + 1;
                $display("  MMIO MISMATCH C[%0d][%0d]: got=%0d exp=%0d",
                         i,j,r,Cref[i][j]);
            end
        end

        $display("");
        $display("================================================");
        $display("  MMIO matmul:  pass=%0d fail=%0d", pass, fail);
        $display("  start->done (bus cycles, incl. polling): %0d", t1 - t0);
        $display("================================================");
        #50 $finish;
    end
endmodule
