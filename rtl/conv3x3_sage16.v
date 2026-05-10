// =============================================================================
// conv3x3_sage16.v  — 3x3 convolution kernel on the 4x4 sage16 fabric
//
// Architecture:
//   * Instantiates 16 unmodified `pe` cores in a generate loop.
//   * Per-PE 9:1 input mux selects the current pixel I[r+di][c+dj] for
//     each output position (r,c) on the current tap.
//   * The kernel weight K[di][dj] is broadcast to all 16 PEs each tap.
//   * Uses OP_MACB_S (signed MAC) — exactly the ISA reconfiguration knob
//     pe.v provides for signed kernels (Sobel, Laplacian, sharpen).
//
// Inputs and outputs are SIGNED 16-bit (pixels) / 16-bit (weights) / 32-bit
// (outputs).  Callers passing unsigned 8-bit pixels simply zero-extend.
//
// Cycle count (start -> done), PIPELINE=1 :
//   S_IDLE -> S_CFG(1) -> S_ACC x 11 (9 taps + 2 DSP drain) -> S_CAP(1)
//   -> S_DONE  = 15 fabric cycles; measured 16 via TB handshake sampling.
// =============================================================================
module conv3x3_sage16 #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1
)(
    input                         clk, rst_n,
    input                         start,
    input  [36*DATA_W-1:0]        img_in,    // 6x6 signed (or zero-ext-unsigned) pixels
    input  [ 9*DATA_W-1:0]        k_in,      // 3x3 signed kernel, row-major
    output [16*ACC_W-1:0]         c_out,     // 4x4 signed output tile
    output reg                    done
);
    // ------------------- opcodes / config -------------------
    // OP_MACB_S is the signed variant of OP_MACB; it uses the exact same
    // DSP48E1 slice and is enabled by a single opcode bit in the PE config.
    localparam [3:0]       OP_MACB_S = 4'd10;
    localparam [CFG_W-1:0] CFG_MACB_S = {OP_MACB_S, 3'd0, 3'd0};

    // ------------------- FSM -------------------
    localparam [2:0] S_IDLE=3'd0, S_CFG=3'd1, S_ACC=3'd2, S_CAP=3'd3, S_DONE=3'd4;
    localparam [3:0] ACC_LAST = PIPELINE ? 4'd10 : 4'd8;

    reg [2:0]        state;
    reg [3:0]        pass_k;
    reg              cfg_broadcast, clr_acc_all, out_en_all;
    reg [CFG_W-1:0]  cfg_data;

    // Tap decode: pass_k in [0..8] -> (di, dj) = (pass_k/3, pass_k%3)
    function [1:0] tap_di; input [3:0] t;
        case(t)
            4'd0,4'd1,4'd2: tap_di = 2'd0;
            4'd3,4'd4,4'd5: tap_di = 2'd1;
            4'd6,4'd7,4'd8: tap_di = 2'd2;
            default:        tap_di = 2'd0;
        endcase
    endfunction
    function [1:0] tap_dj; input [3:0] t;
        case(t)
            4'd0,4'd3,4'd6: tap_dj = 2'd0;
            4'd1,4'd4,4'd7: tap_dj = 2'd1;
            4'd2,4'd5,4'd8: tap_dj = 2'd2;
            default:        tap_dj = 2'd0;
        endcase
    endfunction

    wire [1:0] di_now = tap_di(pass_k);
    wire [1:0] dj_now = tap_dj(pass_k);
    wire       tap_valid = (state == S_ACC) && (pass_k < 4'd9);

    // Kernel weight broadcast to all 16 PEs for the current tap.
    // Directly signed — no bias, no correction.
    wire [DATA_W-1:0] weight_now = tap_valid
                                 ? k_in[(di_now*3 + dj_now)*DATA_W +: DATA_W]
                                 : {DATA_W{1'b0}};

    // ------------------- per-PE array -------------------
    wire [ACC_W-1:0]  pe_acc  [0:15];
    wire [DATA_W-1:0] pe_mesh [0:15];

    genvar r, c;
    generate for(r=0; r<4; r=r+1) begin : rg
        for(c=0; c<4; c=c+1) begin : cg
            wire [5:0] patch_idx = (r + di_now) * 3'd6 + (c + dj_now);
            wire [DATA_W-1:0] pixel_raw =
                img_in[patch_idx*DATA_W +: DATA_W];
            wire [DATA_W-1:0] pixel_now = tap_valid ? pixel_raw
                                                    : {DATA_W{1'b0}};

            pe #(
                .DATA_W  (DATA_W),
                .ACC_W   (ACC_W),
                .CFG_W   (CFG_W),
                .PIPELINE(PIPELINE)
            ) u_pe (
                .clk(clk), .rst_n(rst_n),
                .cfg_load (cfg_broadcast),
                .cfg_in   (cfg_data),
                .out_en   (out_en_all),
                .clr_acc  (clr_acc_all),
                .in_north ({DATA_W{1'b0}}),
                .in_south ({DATA_W{1'b0}}),
                .in_east  ({DATA_W{1'b0}}),
                .in_west  ({DATA_W{1'b0}}),
                .in_self  (pe_acc [r*4+c]),
                .in_bypass(pixel_now),
                .in_b_col (weight_now),
                .out_mesh (pe_mesh[r*4+c]),
                .out      (pe_acc [r*4+c])
            );
        end
    end endgenerate

    // ------------------- output capture -------------------
    reg [ACC_W-1:0] c_reg [0:15];
    genvar gi;
    generate for(gi=0; gi<16; gi=gi+1)
        assign c_out[gi*ACC_W +: ACC_W] = c_reg[gi];
    endgenerate

    // ------------------- FSM sequential -------------------
    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state         <= S_IDLE;
            pass_k        <= 0;
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            out_en_all    <= 0;
            cfg_data      <= 0;
            done          <= 0;
            for(ii=0; ii<16; ii=ii+1) c_reg[ii] <= 0;
        end else begin
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            out_en_all    <= 0;

            case(state)
            S_IDLE: begin
                if(start) begin
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    cfg_data      <= CFG_MACB_S;
                    done          <= 0;
                    pass_k        <= 0;
                    state         <= S_CFG;
                end
            end

            S_CFG: begin
                out_en_all <= 1;
                pass_k     <= 0;
                state      <= S_ACC;
            end

            S_ACC: begin
                out_en_all <= 1;
                if(pass_k == ACC_LAST) begin
                    out_en_all <= 0;
                    state      <= S_CAP;
                end else begin
                    pass_k <= pass_k + 4'd1;
                end
            end

            S_CAP: begin
                for(ii=0; ii<16; ii=ii+1)
                    c_reg[ii] <= pe_acc[ii];
                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1;
                if(start) begin
                    done          <= 0;
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    cfg_data      <= CFG_MACB_S;
                    pass_k        <= 0;
                    state         <= S_CFG;
                end else begin
                    state <= S_IDLE;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
