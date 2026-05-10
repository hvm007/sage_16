// =============================================================================
// sage16_top.v  — SHARED-FABRIC multi-kernel sage16 wrapper
//
// This is the real reconfigurability module: one physical `sage16_4x4_mac`
// instance (16 PEs, 16 DSP48E1s) is driven by a 2-bit `mode` input to
// execute matmul, conv3x3, or quaternion-multiply.  The three kernels
// share the SAME 16 PEs.  No additional fabric is synthesised per kernel.
//
// Mode selection:
//   mode = 2'd0 : matmul_4x4    (unsigned, OP_MACB )
//   mode = 2'd1 : conv3x3       (signed,   OP_MACB_S, per-PE bypass)
//   mode = 2'd2 : quaternion    (signed,   OP_MACB_S)
//   mode = 2'd3 : matmul_4x4 S  (signed,   OP_MACB_S, same routing as MODE_MM)
//                                used for CNN FC layers with signed INT weights
//
// The mode input is latched when `start` is asserted.  While the FSM is
// running, the latched `mode_reg` steers operand delivery, the PE opcode,
// the per-PE bypass enable, and the output layout (quat transposes).
//
// Cycle counts (start -> done, PIPELINE=1):
//   matmul : 11 cycles   (4 real taps + 2 DSP drain + cfg + cap + handshake)
//   quat   : 11 cycles   (same shape as matmul)
//   conv3x3: 16 cycles   (9 real taps + 2 DSP drain + cfg + cap + handshake)
//
// Zero-cycle reconfiguration: the moment `done` deasserts, a new `start`
// with a different `mode` kicks off the next kernel.  No fabric reset,
// no PE reprogramming cost beyond the single broadcast-cfg cycle.
// =============================================================================
module sage16_top #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1
)(
    input                       clk, rst_n,
    input  [1:0]                mode,           // latched at start
    input                       start,
    output reg                  done,
    output [1:0]                mode_out,       // currently-active mode (for TB)

    // --- matmul inputs (unsigned int16, row-major 4x4) ---
    input  [16*DATA_W-1:0]      mm_a,
    input  [16*DATA_W-1:0]      mm_b,

    // --- conv3x3 inputs (signed pixels 16b, signed kernel 16b) ---
    input  [36*DATA_W-1:0]      cv_img,         // 6x6 row-major
    input  [ 9*DATA_W-1:0]      cv_k,           // 3x3 row-major

    // --- quat inputs (signed) ---
    input  signed [DATA_W-1:0]  qt_q1_w, qt_q1_x, qt_q1_y, qt_q1_z,
    input  [16*DATA_W-1:0]      qt_q2,          // 4 quats, [j*4+k]

    // --- unified output ---
    //   matmul : c_out[i*4+j] = C[i][j]       (unsigned)
    //   conv   : c_out[r*4+c] = O[r][c]       (signed)
    //   quat   : c_out[j*4+k] = (q1*q2_j)[k]  (signed) — auto-transposed
    output [16*ACC_W-1:0]       c_out
);
    localparam [1:0] MODE_MM  = 2'd0, MODE_CV = 2'd1, MODE_QT = 2'd2;
    localparam [1:0] MODE_MMS = 2'd3;   // signed matmul; same routing as MM
    localparam [3:0] OP_MACB   = 4'd9;
    localparam [3:0] OP_MACB_S = 4'd10;

    assign mode_out = mode_reg;

    // ---------------- mode register ----------------
    reg [1:0] mode_reg;

    // ---------------- FSM states --------------------
    localparam [2:0] S_IDLE=3'd0, S_CFG=3'd1, S_ACC=3'd2, S_CAP=3'd3, S_DONE=3'd4;
    reg [2:0]        state;
    reg [3:0]        pass_k;

    // ---------------- per-mode ACC_LAST ---------------
    // matmul/quat share the same 4-tap + drain schedule; conv has 9 taps.
    wire [3:0] acc_last =
        (mode_reg == MODE_CV) ? (PIPELINE ? 4'd10 : 4'd8)
                              : (PIPELINE ? 4'd5  : 4'd3);

    // ---------------- per-mode opcode ----------------
    wire [3:0]       op_sel =
        (mode_reg == MODE_MM) ? OP_MACB : OP_MACB_S;
    wire [CFG_W-1:0] cfg_word_new = {op_sel, 3'd0, 3'd0};

    // ---------------- fabric instance ----------------
    reg                       cfg_broadcast, clr_acc_all, out_en_all;
    reg  [CFG_W-1:0]          cfg_data;
    reg  [ 4*DATA_W-1:0]      west_in_r, north_in_r;
    reg  [16*DATA_W-1:0]      per_pe_bypass_flat;
    wire                      per_pe_bypass_en = (mode_reg == MODE_CV);
    wire [16*ACC_W-1:0]       all_pe_out;

    sage16_4x4_mac #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .CFG_W   (CFG_W),
        .PIPELINE(PIPELINE)
    ) u_fab (
        .clk(clk), .rst_n(rst_n),
        .cfg_pe_row(2'd0), .cfg_pe_col(2'd0),
        .cfg_data(cfg_data),
        .cfg_load(1'b0),
        .cfg_broadcast(cfg_broadcast),
        .clr_acc_all(clr_acc_all),
        .out_en_all(out_en_all),
        .ext_in_west (west_in_r),
        .ext_in_north(north_in_r),
        .per_pe_bypass_en  (per_pe_bypass_en),
        .per_pe_bypass_flat(per_pe_bypass_flat),
        .ext_out_east(),
        .all_pe_out  (all_pe_out)
    );

    // ---------------- quaternion Q-matrix (combinational) ----------------
    wire signed [DATA_W-1:0] A_qt [0:3][0:3];
    assign A_qt[0][0] =  qt_q1_w;  assign A_qt[0][1] = -qt_q1_x;
    assign A_qt[0][2] = -qt_q1_y;  assign A_qt[0][3] = -qt_q1_z;
    assign A_qt[1][0] =  qt_q1_x;  assign A_qt[1][1] =  qt_q1_w;
    assign A_qt[1][2] = -qt_q1_z;  assign A_qt[1][3] =  qt_q1_y;
    assign A_qt[2][0] =  qt_q1_y;  assign A_qt[2][1] =  qt_q1_z;
    assign A_qt[2][2] =  qt_q1_w;  assign A_qt[2][3] = -qt_q1_x;
    assign A_qt[3][0] =  qt_q1_z;  assign A_qt[3][1] = -qt_q1_y;
    assign A_qt[3][2] =  qt_q1_x;  assign A_qt[3][3] =  qt_q1_w;

    // ---------------- conv tap decode ----------------
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
    wire [1:0] cv_di = tap_di(pass_k);
    wire [1:0] cv_dj = tap_dj(pass_k);

    // ---------------- operand-drive (combinational, mode-steered) --------
    integer rL, cL;
    reg  [1:0] kk;
    reg  [DATA_W-1:0] cv_weight;
    always @(*) begin
        west_in_r          = { 4*DATA_W {1'b0} };
        north_in_r         = { 4*DATA_W {1'b0} };
        per_pe_bypass_flat = {16*DATA_W {1'b0} };

        if (state == S_ACC) begin
            case (mode_reg)
                // ------ matmul (unsigned or signed): row-broadcast A column,
                //        col-broadcast B row.  MM = unsigned, MMS = signed.
                //        Routing is identical; opcode differs (selected in
                //        S_IDLE/S_DONE via the mode-to-OP_MACB/OP_MACB_S mux).
                MODE_MM, MODE_MMS: if (pass_k < 4'd4) begin
                    kk = pass_k[1:0];
                    for (rL=0; rL<4; rL=rL+1)
                        west_in_r[rL*DATA_W +: DATA_W] =
                            mm_a[(rL*4 + kk)*DATA_W +: DATA_W];
                    for (cL=0; cL<4; cL=cL+1)
                        north_in_r[cL*DATA_W +: DATA_W] =
                            mm_b[(kk*4 + cL)*DATA_W +: DATA_W];
                end

                // ------ quat: same fabric topology, Q-matrix on A, q2 columns on B
                MODE_QT: if (pass_k < 4'd4) begin
                    kk = pass_k[1:0];
                    for (rL=0; rL<4; rL=rL+1)
                        west_in_r[rL*DATA_W +: DATA_W]  = A_qt[rL][kk];
                    // B[k][j] = q2_j[k]; north_in[j] = q2_j[pass_k]
                    for (cL=0; cL<4; cL=cL+1)
                        north_in_r[cL*DATA_W +: DATA_W] =
                            qt_q2[(cL*4 + kk)*DATA_W +: DATA_W];
                end

                // ------ conv: per-PE pixel bypass, kernel broadcast to all cols
                MODE_CV: if (pass_k < 4'd9) begin
                    cv_weight =
                        cv_k[(cv_di*3 + cv_dj)*DATA_W +: DATA_W];
                    for (rL=0; rL<4; rL=rL+1)
                        for (cL=0; cL<4; cL=cL+1)
                            per_pe_bypass_flat[(rL*4 + cL)*DATA_W +: DATA_W] =
                                cv_img[((rL + cv_di)*6 + (cL + cv_dj))*DATA_W +: DATA_W];
                    for (cL=0; cL<4; cL=cL+1)
                        north_in_r[cL*DATA_W +: DATA_W] = cv_weight;
                end

                default: ;
            endcase
        end
    end

    // ---------------- output capture ----------------
    reg [ACC_W-1:0] c_reg [0:15];

    // Two flat views (natural and transposed for quat) — MUX chooses one.
    wire [16*ACC_W-1:0] c_natural;
    wire [16*ACC_W-1:0] c_transposed;
    genvar oi, oj;
    generate for(oi=0; oi<4; oi=oi+1) begin : T
        for(oj=0; oj<4; oj=oj+1) begin : U
            assign c_natural   [(oi*4+oj)*ACC_W +: ACC_W] = c_reg[oi*4+oj];
            assign c_transposed[(oj*4+oi)*ACC_W +: ACC_W] = c_reg[oi*4+oj];
        end
    end endgenerate
    assign c_out = (mode_reg == MODE_QT) ? c_transposed : c_natural;

    // ---------------- FSM sequential ----------------
    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            pass_k        <= 0;
            mode_reg      <= MODE_MM;
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            out_en_all    <= 0;
            cfg_data      <= 0;
            done          <= 0;
            for (ii=0; ii<16; ii=ii+1) c_reg[ii] <= 0;
        end else begin
            cfg_broadcast <= 0;
            clr_acc_all   <= 0;
            out_en_all    <= 0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    mode_reg      <= mode;
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    // Evaluate opcode based on the *incoming* mode (mode_reg
                    // hasn't updated yet this cycle)
                    cfg_data      <= (mode == MODE_MM)
                                     ? {OP_MACB,   3'd0, 3'd0}
                                     : {OP_MACB_S, 3'd0, 3'd0};
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
                if (pass_k == acc_last) begin
                    out_en_all <= 0;
                    state      <= S_CAP;
                end else begin
                    pass_k <= pass_k + 4'd1;
                end
            end

            S_CAP: begin
                for (ii=0; ii<16; ii=ii+1)
                    c_reg[ii] <= all_pe_out[ii*ACC_W +: ACC_W];
                state <= S_DONE;
            end

            S_DONE: begin
                done <= 1;
                if (start) begin
                    // zero-cycle reconfiguration: next kernel starts now
                    mode_reg      <= mode;
                    cfg_broadcast <= 1;
                    clr_acc_all   <= 1;
                    cfg_data      <= (mode == MODE_MM)
                                     ? {OP_MACB,   3'd0, 3'd0}
                                     : {OP_MACB_S, 3'd0, 3'd0};
                    done          <= 0;
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
