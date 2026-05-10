// =============================================================================
// matmul_sage16.v  — FSM-level driver for 4x4 output-stationary matmul
//
// Cycles (from `start` pulse observed by FSM to `done` asserted):
//
//   PIPELINE=0 :  S_IDLE → S_CFG(1) → S_ACC×4 → S_CAP(1) → S_DONE  =  8 cycles
//   PIPELINE=1 :  S_IDLE → S_CFG(1) → S_ACC×6 → S_CAP(1) → S_DONE  = 10 cycles
//                                 (last 2 ACC cycles drain MUL.A + MUL.P pipe regs)
//
// At 100 MHz: 80 ns (PIPELINE=0) / 100 ns (PIPELINE=1).
// PIPELINE=1 closes timing >200 MHz on xc7a35t, so real wall-clock
// drops from 100 ns to ~55 ns despite the extra drain cycle.
//
// Interface preserves DATA_W-wide A/B inputs (row-major 4x4).
// Output C is now ACC_W=32 bits per element — 16 * 32 = 512 bits.
// Testbenches updated accordingly.
// =============================================================================
module matmul_sage16 #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1,
    // Opcode driven to every PE.  Default = OP_MACB (unsigned) for
    // backward compatibility with the original matmul tests; callers
    // (e.g. quat_sage16) can pass OP_MACB_S (=10) to get signed arithmetic
    // using the same fabric.
    parameter [3:0] OPCODE = 4'd9
)(
    input                         clk, rst_n,
    input                         start,
    input  [16*DATA_W-1:0]        a_in, b_in,
    output [16*ACC_W-1:0]         c_out,
    output reg                    done
);
    // --------------------- sage16 fabric ---------------------
    reg  [1:0]             cfg_row, cfg_col;
    reg  [CFG_W-1:0]       cfg_data;
    reg                    cfg_load, cfg_broadcast, clr_acc_all, out_en_all;
    reg  [4*DATA_W-1:0]    west_in_r, north_in_r;
    wire [4*ACC_W-1:0]     east_out;
    wire [16*ACC_W-1:0]    all_pe_out;

    sage16_4x4_mac #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .CFG_W   (CFG_W),
        .PIPELINE(PIPELINE)
    ) u_sage16(
        .clk(clk), .rst_n(rst_n),
        .cfg_pe_row(cfg_row), .cfg_pe_col(cfg_col),
        .cfg_data(cfg_data), .cfg_load(cfg_load),
        .cfg_broadcast(cfg_broadcast),
        .clr_acc_all(clr_acc_all),
        .out_en_all(out_en_all),
        .ext_in_west (west_in_r),
        .ext_in_north(north_in_r),
        .per_pe_bypass_en   (1'b0),        // matmul uses row-broadcast only
        .per_pe_bypass_flat ({16*DATA_W{1'b0}}),
        .ext_out_east(east_out),
        .all_pe_out  (all_pe_out)
    );

    // --------------------- opcodes ---------------------
    // OPCODE is a module-level parameter (default 9 = OP_MACB).
    localparam [CFG_W-1:0] CFG_MACB = {OPCODE, 3'd0, 3'd0};

    // --------------------- index helpers ---------------------
    function [DATA_W-1:0] ae; input [1:0] i,k;
        ae = a_in[(i*4 + k)*DATA_W +: DATA_W];
    endfunction
    function [DATA_W-1:0] be; input [1:0] k,j;
        be = b_in[(k*4 + j)*DATA_W +: DATA_W];
    endfunction

    // --------------------- result register ---------------------
    reg [ACC_W-1:0] c_reg [0:15];
    genvar gi;
    generate for(gi=0; gi<16; gi=gi+1)
        assign c_out[gi*ACC_W +: ACC_W] = c_reg[gi];
    endgenerate

    // --------------------- FSM ---------------------
    localparam [2:0] S_IDLE = 3'd0,
                     S_CFG  = 3'd1,
                     S_ACC  = 3'd2,
                     S_CAP  = 3'd3,
                     S_DONE = 3'd4;

    // PIPELINE=1 adds A/B input regs + P output reg inside the DSP; each
    // adds one cycle of drain latency, so ACC_LAST grows by 2.
    localparam [2:0] ACC_LAST = PIPELINE ? 3'd5 : 3'd3;

    reg [2:0] state;
    reg [2:0] pass_k;
    integer   ii;

    // Combinational drive of row/col broadcast inputs. While pass_k < 4 we
    // stream real data; when pass_k == 4 (drain) we drive zeros so the
    // MUL pipe register flushes cleanly.
    always @(*) begin
        west_in_r  = 0;
        north_in_r = 0;
        if(state == S_ACC && pass_k < 3'd4) begin
            west_in_r [0*DATA_W +: DATA_W] = ae(2'd0, pass_k[1:0]);
            west_in_r [1*DATA_W +: DATA_W] = ae(2'd1, pass_k[1:0]);
            west_in_r [2*DATA_W +: DATA_W] = ae(2'd2, pass_k[1:0]);
            west_in_r [3*DATA_W +: DATA_W] = ae(2'd3, pass_k[1:0]);
            north_in_r[0*DATA_W +: DATA_W] = be(pass_k[1:0], 2'd0);
            north_in_r[1*DATA_W +: DATA_W] = be(pass_k[1:0], 2'd1);
            north_in_r[2*DATA_W +: DATA_W] = be(pass_k[1:0], 2'd2);
            north_in_r[3*DATA_W +: DATA_W] = be(pass_k[1:0], 2'd3);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            state         <= S_IDLE;
            pass_k        <= 0;
            cfg_load      <= 0;
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            cfg_row       <= 0;
            cfg_col       <= 0;
            cfg_data      <= 0;
            out_en_all    <= 0;
            done          <= 0;
            for(ii=0; ii<16; ii=ii+1) c_reg[ii] <= 0;
        end else begin
            cfg_load      <= 0;
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            out_en_all    <= 0;

            case(state)
            S_IDLE: begin
                if(start) begin
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    cfg_data      <= CFG_MACB;
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
                    pass_k <= pass_k + 3'd1;
                end
            end

            S_CAP: begin
                for(ii=0; ii<16; ii=ii+1)
                    c_reg[ii] <= all_pe_out[ii*ACC_W +: ACC_W];
                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1;
                if(start) begin
                    done          <= 0;
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    cfg_data      <= CFG_MACB;
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
