`timescale 1ns/1ps
module tb_matmul;
    parameter DATA_W = 16;
    parameter ACC_W  = 32;
    reg  clk, rst_n, start;
    reg  [16*DATA_W-1:0] a_in, b_in;
    wire [16*ACC_W-1:0]  c_out;
    wire done;

    matmul_sage16 #(.DATA_W(DATA_W), .ACC_W(ACC_W)) dut(
        .clk(clk), .rst_n(rst_n), .start(start),
        .a_in(a_in), .b_in(b_in),
        .c_out(c_out), .done(done)
    );

    initial clk = 0;
    always #5 clk = ~clk;        // 100 MHz

    integer i, j, k;
    integer pass, fail;
    integer total_pass, total_fail;
    integer cyc_start, cyc_end, cycles;
    reg [31:0] cyc_cnt;
    reg [DATA_W-1:0] A[0:3][0:3], B[0:3][0:3];
    reg [ACC_W-1:0]  C_ref[0:3][0:3];
    reg [ACC_W-1:0]  sage16_val, ref_val;

    // free-running cycle counter
    always @(posedge clk or negedge rst_n)
        if(!rst_n) cyc_cnt <= 0;
        else       cyc_cnt <= cyc_cnt + 1;

    task load_ab;
        integer ii, jj;
        begin
            for(ii=0; ii<4; ii=ii+1) for(jj=0; jj<4; jj=jj+1) begin
                a_in[(ii*4+jj)*DATA_W +: DATA_W] = A[ii][jj];
                b_in[(ii*4+jj)*DATA_W +: DATA_W] = B[ii][jj];
            end
        end
    endtask

    task ref_mul;
        integer ri, rj, rk;
        begin
            for(ri=0; ri<4; ri=ri+1) for(rj=0; rj<4; rj=rj+1) begin
                C_ref[ri][rj] = 0;
                for(rk=0; rk<4; rk=rk+1)
                    C_ref[ri][rj] = C_ref[ri][rj] + A[ri][rk] * B[rk][rj];
            end
        end
    endtask

    task run_one;
        begin
            @(posedge clk); #1;
            cyc_start = cyc_cnt;
            start = 1;
            @(posedge clk); #1;
            start = 0;
            wait(done);
            @(posedge clk); #1;
            cyc_end = cyc_cnt;
            cycles  = cyc_end - cyc_start;
        end
    endtask

    task check_and_print;
        input [127:0] test_name;
        integer ci, cj;
        begin
            pass = 0; fail = 0;
            for(ci=0; ci<4; ci=ci+1) for(cj=0; cj<4; cj=cj+1) begin
                sage16_val = c_out[(ci*4+cj)*ACC_W +: ACC_W];
                ref_val  = C_ref[ci][cj];
                if(sage16_val == ref_val) pass = pass + 1;
                else begin
                    fail = fail + 1;
                    $display("  MISMATCH C[%0d][%0d]: got=%0d exp=%0d",
                             ci, cj, sage16_val, ref_val);
                end
            end
            $display("%s: pass=%0d fail=%0d  cycles=%0d (start->done)",
                     test_name, pass, fail, cycles);
            total_pass = total_pass + pass;
            total_fail = total_fail + fail;
        end
    endtask

    initial begin
        $dumpfile("build/matmul_tb.vcd");
        $dumpvars(0, tb_matmul);
        clk = 0; rst_n = 0; start = 0; a_in = 0; b_in = 0;
        total_pass = 0; total_fail = 0;

        // global reset
        #23 rst_n = 1;
        @(posedge clk); #1;

        // ---------- TEST 1: I * I = I ----------
        A[0][0]=1;A[0][1]=0;A[0][2]=0;A[0][3]=0;
        A[1][0]=0;A[1][1]=1;A[1][2]=0;A[1][3]=0;
        A[2][0]=0;A[2][1]=0;A[2][2]=1;A[2][3]=0;
        A[3][0]=0;A[3][1]=0;A[3][2]=0;A[3][3]=1;
        B[0][0]=1;B[0][1]=0;B[0][2]=0;B[0][3]=0;
        B[1][0]=0;B[1][1]=1;B[1][2]=0;B[1][3]=0;
        B[2][0]=0;B[2][1]=0;B[2][2]=1;B[2][3]=0;
        B[3][0]=0;B[3][1]=0;B[3][2]=0;B[3][3]=1;
        load_ab; ref_mul;
        run_one;
        check_and_print("TEST1 I*I      ");

        // ---------- TEST 2: 2x2 block non-trivial ----------
        A[0][0]=1;A[0][1]=2;A[0][2]=0;A[0][3]=0;
        A[1][0]=3;A[1][1]=4;A[1][2]=0;A[1][3]=0;
        A[2][0]=0;A[2][1]=0;A[2][2]=1;A[2][3]=0;
        A[3][0]=0;A[3][1]=0;A[3][2]=0;A[3][3]=1;
        B[0][0]=5;B[0][1]=6;B[0][2]=0;B[0][3]=0;
        B[1][0]=7;B[1][1]=8;B[1][2]=0;B[1][3]=0;
        B[2][0]=0;B[2][1]=0;B[2][2]=2;B[2][3]=0;
        B[3][0]=0;B[3][1]=0;B[3][2]=0;B[3][3]=3;
        load_ab; ref_mul;
        run_one;
        check_and_print("TEST2 block    ");

        // ---------- TEST 3: Dense ----------
        A[0][0]=2;A[0][1]=3;A[0][2]=1;A[0][3]=4;
        A[1][0]=5;A[1][1]=1;A[1][2]=2;A[1][3]=3;
        A[2][0]=0;A[2][1]=4;A[2][2]=3;A[2][3]=5;
        A[3][0]=6;A[3][1]=2;A[3][2]=2;A[3][3]=8;
        B[0][0]=1;B[0][1]=2;B[0][2]=3;B[0][3]=4;
        B[1][0]=5;B[1][1]=6;B[1][2]=7;B[1][3]=8;
        B[2][0]=9;B[2][1]=1;B[2][2]=2;B[2][3]=3;
        B[3][0]=4;B[3][1]=5;B[3][2]=6;B[3][3]=7;
        load_ab; ref_mul;
        run_one;
        check_and_print("TEST3 dense    ");

        // ---------- TEST 4: Randomized stress 8-bit (10 iters) ----------
        begin : rand_tests
            integer t, ri, rj;
            for(t=0; t<10; t=t+1) begin
                for(ri=0; ri<4; ri=ri+1) for(rj=0; rj<4; rj=rj+1) begin
                    A[ri][rj] = $random & 16'h00FF;
                    B[ri][rj] = $random & 16'h00FF;
                end
                load_ab; ref_mul;
                run_one;
                check_and_print("TEST4 random8  ");
            end
        end

        // ---------- TEST 5: Overflow proof — needs the 32-bit accumulator ----------
        // All A=255, all B=255.  Each C element = 4 * 255 * 255 = 260100
        // This is > 65535 (2^16), so a 16-bit accumulator would give 63556.
        begin : overflow_test
            integer ri, rj;
            for(ri=0; ri<4; ri=ri+1) for(rj=0; rj<4; rj=rj+1) begin
                A[ri][rj] = 16'h00FF;
                B[ri][rj] = 16'h00FF;
            end
            load_ab; ref_mul;
            run_one;
            check_and_print("TEST5 >16b     ");
            $display("  (each C[i][j] must equal 260100 — proves ACC_W=32 works)");
        end

        // ---------- TEST 6: 16-bit full-range stress ----------
        begin : big_rand
            integer t, ri, rj;
            for(t=0; t<5; t=t+1) begin
                for(ri=0; ri<4; ri=ri+1) for(rj=0; rj<4; rj=rj+1) begin
                    A[ri][rj] = $random;          // full 16-bit
                    B[ri][rj] = $random;
                end
                load_ab; ref_mul;
                run_one;
                check_and_print("TEST6 rand16   ");
            end
        end

        $display("\n================================================");
        $display("  SUMMARY: pass=%0d fail=%0d", total_pass, total_fail);
        $display("================================================");
        #50 $finish;
    end
endmodule
