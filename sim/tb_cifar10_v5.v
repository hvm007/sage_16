`timescale 1ns/1ps
// =============================================================================
// tb_cifar10_v5.v -- CIFAR-10 v5 smoke inference on sage16_top
//
// Head-start RTL smoke path for the frozen CIFAR-v5 fixed/export candidate:
//   3x32x32 -> Conv3x3(3->64,pad1)+ReLU -> Conv3x3(64->64,pad1)+ReLU
//   -> AvgPool2 -> Conv3x3(64->128,pad1)+ReLU -> Conv3x3(128->128,pad1)+ReLU
//   -> AvgPool2 -> Conv3x3(128->128,pad1)+ReLU -> Conv3x3(128->128,pad1)+ReLU
//   -> GlobalAvgPool8x8 -> >>>4 -> FC(128->10)
//
// BatchNorm is already folded in the export. This testbench consumes the folded
// int16 conv weights, signed int32 conv biases, int16 FC weights, images,
// labels, and preds from data/cifar10_v5/.
//
// Compile with +DREAL_CIFAR10_V5 to load real artifacts. Override N_IMG with
// -Ptb_cifar10_v5.N_IMG=<small N>; default is 2.
//
// Current scope:
// - Uses sage16_top mode=1 for conv3x3 and mode=3 for signed matmul.
// - Handles 32-bit non-negative activations by splitting them into signed
//   16-bit low/high words before feeding conv/FC primitives.
// - Compares final fabric-scheduled predictions with exported preds.txt.
// - Does not yet include a separate full per-layer software golden checker like
//   tb_cifar10_v2.v; exported preds are the end-to-end oracle for this skeleton.
// =============================================================================
module tb_cifar10_v5;
    parameter DATA_W      = 16;
    parameter ACC_W       = 32;
    parameter N_IMG       = 2;
    parameter POOL1_SHIFT = 0;
    parameter POOL2_SHIFT = 0;
    parameter GAP_SHIFT   = 0;
    parameter FINAL_SHIFT = 4;

    localparam HIN   = 32;
    localparam MAX_H = 32;
    localparam MAX_C = 128;
    localparam ACT_N = MAX_C*MAX_H*MAX_H;
    localparam FC_IN = 128;
    localparam NCLS  = 12;

    reg clk = 0; always #5 clk = ~clk;

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

    integer cyc_cnt;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) cyc_cnt <= 0; else cyc_cnt <= cyc_cnt + 1;

    reg done_d;
    wire done_rise = done & ~done_d;
    always @(posedge clk) done_d <= done;

    integer layer_cycles;
    integer cy_conv_tot, cy_pool_tot, cy_gap_tot, cy_fc_tot;
    integer cy_conv_i, cy_pool_i, cy_gap_i, cy_fc_i;
    integer n_pass_pred, n_fail_pred, n_correct_label, n_total_label;
    integer i, j, ch, r, c, oc, ic, img, rr, cc, kk, ii, jj, limb;
    integer argmax_i;
    reg signed [63:0] amax_v;

    reg signed [DATA_W-1:0] IMG     [0:N_IMG*3*HIN*HIN-1];
    reg signed [DATA_W-1:0] CONV1_K [0:64*3*9-1];
    reg signed [DATA_W-1:0] CONV2_K [0:64*64*9-1];
    reg signed [DATA_W-1:0] CONV3_K [0:128*64*9-1];
    reg signed [DATA_W-1:0] CONV4_K [0:128*128*9-1];
    reg signed [DATA_W-1:0] CONV5_K [0:128*128*9-1];
    reg signed [DATA_W-1:0] CONV6_K [0:128*128*9-1];
    reg signed [ACC_W-1:0]  CONV1_B [0:63];
    reg signed [ACC_W-1:0]  CONV2_B [0:63];
    reg signed [ACC_W-1:0]  CONV3_B [0:127];
    reg signed [ACC_W-1:0]  CONV4_B [0:127];
    reg signed [ACC_W-1:0]  CONV5_B [0:127];
    reg signed [ACC_W-1:0]  CONV6_B [0:127];
    reg signed [DATA_W-1:0] W1      [0:10*FC_IN-1];
    reg signed [DATA_W-1:0] LABEL   [0:N_IMG-1];
    reg signed [DATA_W-1:0] PRED    [0:N_IMG-1];

    reg signed [63:0]      ACT_A [0:ACT_N-1];
    reg signed [63:0]      ACT_B [0:ACT_N-1];
    reg signed [63:0]      FEAT  [0:FC_IN-1];
    reg signed [63:0]      LOGIT [0:NCLS-1];

    reg [36*DATA_W-1:0] img_pack;
    reg [ 9*DATA_W-1:0] k_pack;
    reg [16*ACC_W-1:0]  c_pack;
    reg [16*DATA_W-1:0] A_tile, B_tile;
    reg signed [63:0]   tile_acc64 [0:15];

    function integer idx_img;
        input integer n, ch_i, r_i, c_i;
        begin idx_img = (((n*3 + ch_i)*HIN + r_i)*HIN + c_i); end
    endfunction
    function integer idx_act;
        input integer ch_i, r_i, c_i;
        begin idx_act = ((ch_i*MAX_H + r_i)*MAX_H + c_i); end
    endfunction

    function integer idx_conv_k;
        input integer layer_i, oc_i, ic_i, r_i, c_i;
        begin
            case (layer_i)
                1: idx_conv_k = (((oc_i*3   + ic_i)*3 + r_i)*3 + c_i);
                2: idx_conv_k = (((oc_i*64  + ic_i)*3 + r_i)*3 + c_i);
                3: idx_conv_k = (((oc_i*64  + ic_i)*3 + r_i)*3 + c_i);
                4: idx_conv_k = (((oc_i*128 + ic_i)*3 + r_i)*3 + c_i);
                5: idx_conv_k = (((oc_i*128 + ic_i)*3 + r_i)*3 + c_i);
                6: idx_conv_k = (((oc_i*128 + ic_i)*3 + r_i)*3 + c_i);
                default: idx_conv_k = 0;
            endcase
        end
    endfunction

    function signed [DATA_W-1:0] get_k;
        input integer layer_i, oc_i, ic_i, r_i, c_i;
        begin
            case (layer_i)
                1: get_k = CONV1_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                2: get_k = CONV2_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                3: get_k = CONV3_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                4: get_k = CONV4_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                5: get_k = CONV5_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                6: get_k = CONV6_K[idx_conv_k(layer_i, oc_i, ic_i, r_i, c_i)];
                default: get_k = 0;
            endcase
        end
    endfunction

    function signed [ACC_W-1:0] get_b;
        input integer layer_i, oc_i;
        begin
            case (layer_i)
                1: get_b = CONV1_B[oc_i];
                2: get_b = CONV2_B[oc_i];
                3: get_b = CONV3_B[oc_i];
                4: get_b = CONV4_B[oc_i];
                5: get_b = CONV5_B[oc_i];
                6: get_b = CONV6_B[oc_i];
                default: get_b = 0;
            endcase
        end
    endfunction

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

    task split32;
        input signed [ACC_W-1:0] x;
        output signed [DATA_W-1:0] lo16;
        output signed [DATA_W-1:0] hi16;
        reg signed [ACC_W-1:0] hi_part;
        begin
            lo16 = x[DATA_W-1:0];
            hi_part = (x - lo16) >>> 16;
            hi16 = hi_part[DATA_W-1:0];
        end
    endtask

    task split64_limb;
        input signed [63:0] x;
        input integer limb_i;
        output signed [DATA_W-1:0] part16;
        reg signed [63:0] rem;
        reg signed [DATA_W-1:0] part_tmp;
        integer li;
        begin
            rem = x;
            part16 = 0;
            for (li=0; li<=limb_i; li=li+1) begin
                part_tmp = rem[DATA_W-1:0];
                if (li == limb_i) part16 = part_tmp;
                rem = (rem - part_tmp) >>> DATA_W;
            end
        end
    endtask

    task init_data;
        integer s;
        begin
`ifdef REAL_CIFAR10_V5
            $readmemh("data/cifar10_v5/conv1_k.mem", CONV1_K);
            $readmemh("data/cifar10_v5/conv2_k.mem", CONV2_K);
            $readmemh("data/cifar10_v5/conv3_k.mem", CONV3_K);
            $readmemh("data/cifar10_v5/conv4_k.mem", CONV4_K);
            $readmemh("data/cifar10_v5/conv5_k.mem", CONV5_K);
            $readmemh("data/cifar10_v5/conv6_k.mem", CONV6_K);
            $readmemh("data/cifar10_v5/conv1_b.mem", CONV1_B);
            $readmemh("data/cifar10_v5/conv2_b.mem", CONV2_B);
            $readmemh("data/cifar10_v5/conv3_b.mem", CONV3_B);
            $readmemh("data/cifar10_v5/conv4_b.mem", CONV4_B);
            $readmemh("data/cifar10_v5/conv5_b.mem", CONV5_B);
            $readmemh("data/cifar10_v5/conv6_b.mem", CONV6_B);
            $readmemh("data/cifar10_v5/w1.mem",      W1);
            $readmemh("data/cifar10_v5/images.mem",  IMG);
            $readmemh("data/cifar10_v5/labels.mem",  LABEL);
            $readmemh("data/cifar10_v5/preds.txt",   PRED);
            $display("  [REAL_CIFAR10_V5] loaded frozen CIFAR-v5 .mem artifacts, N_IMG=%0d", N_IMG);
`else
            s = 1;
            for (i=0; i<64*3*9; i=i+1) begin CONV1_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<64*64*9; i=i+1) begin CONV2_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<128*64*9; i=i+1) begin CONV3_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<128*128*9; i=i+1) begin CONV4_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<128*128*9; i=i+1) begin CONV5_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<128*128*9; i=i+1) begin CONV6_K[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<64; i=i+1) begin CONV1_B[i] = (s % 17) - 8; CONV2_B[i] = (s % 17) - 8; s=s+1; end
            for (i=0; i<128; i=i+1) begin CONV3_B[i] = (s % 17) - 8; CONV4_B[i] = (s % 17) - 8; CONV5_B[i] = (s % 17) - 8; CONV6_B[i] = (s % 17) - 8; s=s+1; end
            for (i=0; i<10*FC_IN; i=i+1) begin W1[i] = (s % 7) - 3; s=s+1; end
            for (i=0; i<N_IMG*3*HIN*HIN; i=i+1) begin IMG[i] = (s % 64) - 32; s=s+1; end
            for (i=0; i<N_IMG; i=i+1) begin LABEL[i] = 0; PRED[i] = 0; end
`endif
        end
    endtask

    task load_image;
        input integer img_idx;
        begin
            for (i=0; i<ACT_N; i=i+1) begin
                ACT_A[i] = 0;
                ACT_B[i] = 0;
            end
            for (ch=0; ch<3; ch=ch+1)
                for (r=0; r<HIN; r=r+1)
                    for (c=0; c<HIN; c=c+1)
                        ACT_A[idx_act(ch,r,c)] = IMG[idx_img(img_idx,ch,r,c)];
        end
    endtask

    task copy_b_to_a;
        input integer out_ch, h;
        begin
            for (ch=0; ch<out_ch; ch=ch+1)
                for (r=0; r<h; r=r+1)
                    for (c=0; c<h; c=c+1)
                        ACT_A[idx_act(ch,r,c)] = ACT_B[idx_act(ch,r,c)];
        end
    endtask

    task run_conv_layer;
        input integer layer_i, in_ch, out_ch, h;
        integer tr_out, tc_out, ic_l, sr, sc;
        reg signed [63:0] pix;
        reg signed [DATA_W-1:0] part16;
        reg signed [ACC_W-1:0] cur32;
        reg signed [63:0] cur64;
        begin
            layer_cycles = 0;
            for (i=0; i<ACT_N; i=i+1) ACT_B[i] = 0;

            for (oc=0; oc<out_ch; oc=oc+1)
                for (ic_l=0; ic_l<in_ch; ic_l=ic_l+1)
                    for (tr_out=0; tr_out<h; tr_out=tr_out+4)
                        for (tc_out=0; tc_out<h; tc_out=tc_out+4) begin
                            for (i=0; i<3; i=i+1)
                                for (j=0; j<3; j=j+1)
                                    k_pack[(i*3+j)*DATA_W +: DATA_W] = get_k(layer_i, oc, ic_l, i, j);

                            for (limb=0; limb<4; limb=limb+1) begin
                                for (rr=0; rr<6; rr=rr+1)
                                    for (cc=0; cc<6; cc=cc+1) begin
                                        sr = tr_out + rr - 1;
                                        sc = tc_out + cc - 1;
                                        pix = (sr >= 0 && sr < h && sc >= 0 && sc < h) ? ACT_A[idx_act(ic_l,sr,sc)] : 0;
                                        split64_limb(pix, limb, part16);
                                        img_pack[(rr*6+cc)*DATA_W +: DATA_W] = part16;
                                    end
                                run_conv_call(img_pack, k_pack, c_pack);
                                for (rr=0; rr<4; rr=rr+1)
                                    for (cc=0; cc<4; cc=cc+1)
                                        if ((tr_out+rr) < h && (tc_out+cc) < h) begin
                                            cur32 = $signed(c_pack[(rr*4+cc)*ACC_W +: ACC_W]);
                                            cur64 = {{32{cur32[31]}}, cur32};
                                            ACT_B[idx_act(oc,tr_out+rr,tc_out+cc)] =
                                                ACT_B[idx_act(oc,tr_out+rr,tc_out+cc)] + (cur64 <<< (DATA_W*limb));
                                        end
                            end
                        end

            for (oc=0; oc<out_ch; oc=oc+1)
                for (r=0; r<h; r=r+1)
                    for (c=0; c<h; c=c+1) begin
                        ACT_B[idx_act(oc,r,c)] = ACT_B[idx_act(oc,r,c)] + get_b(layer_i, oc);
                        if (ACT_B[idx_act(oc,r,c)] < 0) ACT_B[idx_act(oc,r,c)] = 0;
                    end
            copy_b_to_a(out_ch, h);
            cy_conv_i = layer_cycles;
        end
    endtask

    task run_pool2;
        input integer channels, h_in, shift, h_out;
        integer pi_l, pj_l, t0_sw;
        reg signed [63:0] ps;
        begin
            t0_sw = cyc_cnt;
            for (i=0; i<ACT_N; i=i+1) ACT_B[i] = 0;
            for (oc=0; oc<channels; oc=oc+1)
                for (r=0; r<h_out; r=r+1)
                    for (c=0; c<h_out; c=c+1) begin
                        ps = 0;
                        for (pi_l=0; pi_l<2; pi_l=pi_l+1)
                            for (pj_l=0; pj_l<2; pj_l=pj_l+1)
                                ps = ps + ACT_A[idx_act(oc,r*2+pi_l,c*2+pj_l)];
                        ps = ps >>> 2;
                        if (shift > 0) ps = ps >>> shift;
                        ACT_B[idx_act(oc,r,c)] = ps;
                    end
            copy_b_to_a(channels, h_out);
            repeat (1) @(posedge clk);
            cy_pool_i = cyc_cnt - t0_sw;
        end
    endtask

    task run_gap;
        input integer channels, h, shift;
        reg signed [63:0] ps;
        integer t0_sw;
        begin
            t0_sw = cyc_cnt;
            for (oc=0; oc<channels; oc=oc+1) begin
                ps = 0;
                for (r=0; r<h; r=r+1)
                    for (c=0; c<h; c=c+1)
                        ps = ps + ACT_A[idx_act(oc,r,c)];
                ps = ps >>> 6;
                if (shift > 0) ps = ps >>> shift;
                if (FINAL_SHIFT > 0) ps = ps >>> FINAL_SHIFT;
                FEAT[oc] = ps;
            end
            repeat (1) @(posedge clk);
            cy_gap_i = cyc_cnt - t0_sw;
        end
    endtask

    task run_fc;
        integer oc0;
        reg signed [ACC_W-1:0] sum;
        reg signed [DATA_W-1:0] part16;
        reg signed [63:0] sum64;
        begin
            layer_cycles = 0;
            for (i=0; i<NCLS; i=i+1) LOGIT[i] = 0;
            for (oc0=0; oc0<NCLS; oc0=oc0+4) begin
                for (ii=0; ii<4; ii=ii+1) tile_acc64[ii] = 0;
                for (kk=0; kk<FC_IN; kk=kk+4) begin
                    for (ii=0; ii<4; ii=ii+1)
                        for (jj=0; jj<4; jj=jj+1)
                            A_tile[(ii*4+jj)*DATA_W +: DATA_W] =
                                ((oc0+ii) < 10) ? W1[(oc0+ii)*FC_IN + kk+jj] : 0;

                    for (limb=0; limb<4; limb=limb+1) begin
                        for (ii=0; ii<4; ii=ii+1) begin
                            split64_limb(FEAT[kk+ii], limb, part16);
                            B_tile[(ii*4+0)*DATA_W +: DATA_W] = part16;
                            B_tile[(ii*4+1)*DATA_W +: DATA_W] = 0;
                            B_tile[(ii*4+2)*DATA_W +: DATA_W] = 0;
                            B_tile[(ii*4+3)*DATA_W +: DATA_W] = 0;
                        end
                        run_matmul_call(A_tile, B_tile, c_pack);
                        for (ii=0; ii<4; ii=ii+1) begin
                            sum = $signed(c_pack[(ii*4+0)*ACC_W +: ACC_W]);
                            sum64 = {{32{sum[31]}}, sum};
                            tile_acc64[ii] = tile_acc64[ii] + (sum64 <<< (DATA_W*limb));
                        end
                    end
                end
                for (ii=0; ii<4; ii=ii+1) LOGIT[oc0+ii] = tile_acc64[ii];
            end
            cy_fc_i = layer_cycles;
        end
    endtask

    task print_mismatch_diag;
        input integer img_idx;
        begin
            if (img_idx == 0) begin
                $display("  [diag] logits got: c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d c5=%0d c6=%0d c7=%0d c8=%0d c9=%0d",
                         LOGIT[0], LOGIT[1], LOGIT[2], LOGIT[3], LOGIT[4],
                         LOGIT[5], LOGIT[6], LOGIT[7], LOGIT[8], LOGIT[9]);
                $display("  [diag] feat[0..7]=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                         FEAT[0], FEAT[1], FEAT[2], FEAT[3],
                         FEAT[4], FEAT[5], FEAT[6], FEAT[7]);
            end
        end
    endtask

    task check_pred;
        input integer img_idx;
        begin
            argmax_i = 0; amax_v = LOGIT[0];
            for (i=1; i<10; i=i+1)
                if (LOGIT[i] > amax_v) begin amax_v = LOGIT[i]; argmax_i = i; end
            if (argmax_i == PRED[img_idx]) n_pass_pred=n_pass_pred+1;
            else begin
                n_fail_pred=n_fail_pred+1;
                $display("  [pred] img=%0d got=%0d exported=%0d label=%0d",
                         img_idx, argmax_i, PRED[img_idx], LABEL[img_idx]);
                print_mismatch_diag(img_idx);
            end
            n_total_label = n_total_label + 1;
            if (argmax_i == LABEL[img_idx]) n_correct_label = n_correct_label + 1;
        end
    endtask

    initial begin
        rst_n = 0; mode = 0; start = 0;
        mm_a = 0; mm_b = 0; cv_img_r = 0; cv_k_r = 0;
        cy_conv_tot=0; cy_pool_tot=0; cy_gap_tot=0; cy_fc_tot=0;
        n_pass_pred=0; n_fail_pred=0; n_correct_label=0; n_total_label=0;

        init_data;
        #23 rst_n = 1;
        @(posedge clk); #1;

        $display("\n================================================");
        $display("  CIFAR-10 v5 smoke inference on sage16_top");
        $display("  topology: six padded conv3x3 layers, pool/pool/gap, FC 128->10");
        $display("  shifts: pool1>>>%0d pool2>>>%0d gap>>>%0d final>>>%0d",
                 POOL1_SHIFT, POOL2_SHIFT, GAP_SHIFT, FINAL_SHIFT);
        $display("  %0d images; compares fabric-scheduled prediction vs exported preds.txt", N_IMG);
        $display("================================================");

        for (img=0; img<N_IMG; img=img+1) begin
            load_image(img);

            run_conv_layer(1, 3,   64,  32); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_conv_layer(2, 64,  64,  32); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_pool2(64, 32, POOL1_SHIFT, 16); cy_pool_tot = cy_pool_tot + cy_pool_i;
            run_conv_layer(3, 64,  128, 16); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_conv_layer(4, 128, 128, 16); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_pool2(128, 16, POOL2_SHIFT, 8); cy_pool_tot = cy_pool_tot + cy_pool_i;
            run_conv_layer(5, 128, 128, 8); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_conv_layer(6, 128, 128, 8); cy_conv_tot = cy_conv_tot + cy_conv_i;
            run_gap(128, 8, GAP_SHIFT); cy_gap_tot = cy_gap_tot + cy_gap_i;
            run_fc; cy_fc_tot = cy_fc_tot + cy_fc_i;
            check_pred(img);

            $display("  img %0d : pred=%0d label=%0d exported=%0d conv_cycles_so_far=%0d fc=%0d",
                     img, argmax_i, LABEL[img], PRED[img], cy_conv_tot, cy_fc_i);
        end

        $display("");
        $display("================================================");
        $display("  CIFAR-10 v5 SUMMARY (totals over %0d images)", N_IMG);
        $display("================================================");
        $display("  fabric compute cycles : conv=%0d fc=%0d", cy_conv_tot, cy_fc_tot);
        $display("  host SW cycles        : pool=%0d gap=%0d", cy_pool_tot, cy_gap_tot);
        $display("  pred matches export   : %0d / %0d", n_pass_pred, N_IMG);
        $display("  CIFAR top-1 subset    : %0d / %0d", n_correct_label, n_total_label);
        $display("  SUMMARY: pass=%0d fail=%0d", n_pass_pred, n_fail_pred);
        $display("================================================");
        #100 $finish;
    end
endmodule
