// =============================================================================
// pe.v  — Processing Element
//
// Parameters
//   DATA_W    : width of mesh / matrix-element ports (default 16).
//   ACC_W     : width of accumulator / out register (default 32 = 2*DATA_W).
//               Prevents truncation during MAC (16x16 product needs 32b; 4
//               accumulations need 34b, so ACC_W=32 is safe for 4-deep int16
//               accumulation with absolute values <= 2^15).
//   CFG_W     : width of config word (default 10).
//   PIPELINE  : 0 = single-cycle MAC (original behaviour).
//               1 = DSP-friendly: two pipeline registers — one on the
//                   multiplier INPUT operands (absorbed into DSP48E1
//                   AREG/BREG) and one on the multiplier OUTPUT (absorbed
//                   into PREG).  Closes timing >200 MHz on xc7a35t.
//                   Adds 2 cycles of drain latency in the FSM.
//
// Outputs
//   out       : ACC_W-bit accumulator (also drives in_self feedback)
//   out_mesh  : DATA_W-bit truncated view for routing to mesh neighbours
// =============================================================================
module pe #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1
)(
    input                       clk, rst_n,
    input                       cfg_load,
    input      [CFG_W-1:0]      cfg_in,
    input                       out_en,
    input                       clr_acc,
    input      [DATA_W-1:0]     in_north, in_south, in_east, in_west,
    input      [ACC_W-1:0]      in_self,   // full-width self feedback (for MAC)
    input      [DATA_W-1:0]     in_bypass, // row broadcast A[i][k]
    input      [DATA_W-1:0]     in_b_col,  // col broadcast B[k][j]
    output     [DATA_W-1:0]     out_mesh,  // narrow mesh output (lower DATA_W of out)
    output reg [ACC_W-1:0]      out        // accumulator / readout
);
    localparam OP_ADD   =4'd0,  OP_SUB  =4'd1,  OP_MUL  =4'd2, OP_AND =4'd3,
               OP_OR    =4'd4,  OP_XOR  =4'd5,  OP_PASS =4'd6, OP_MUX =4'd7,
               OP_MAC   =4'd8,   // (mesh_a   * mesh_b ) + self   unsigned
               OP_MACB  =4'd9,   // (bypass   * b_col  ) + self   unsigned
               OP_MACB_S=4'd10;  // (bypass   * b_col  ) + self   SIGNED
                                 //   used by conv3x3 (negative kernels) and
                                 //   quat (negative Q-matrix entries). Same
                                 //   DSP48E1, just signed interpretation —
                                 //   one ISA-level reconfiguration bit.

    localparam SEL_N=3'd0, SEL_S=3'd1, SEL_E=3'd2, SEL_W=3'd3, SEL_SELF=3'd4;

    // ------------- config register -------------
    reg [CFG_W-1:0] cfg_reg;
    wire [3:0] op    = cfg_reg[9:6];
    wire [2:0] sel_a = cfg_reg[5:3];
    wire [2:0] sel_b = cfg_reg[2:0];

    always @(posedge clk or negedge rst_n)
        if(!rst_n)         cfg_reg <= 0;
        else if(cfg_load)  cfg_reg <= cfg_in;

    // ------------- mesh-operand mux (DATA_W) -------------
    function [DATA_W-1:0] mux_in;
        input [2:0] s;
        input [DATA_W-1:0] n, so, e, w, sf;
        case(s)
            SEL_N:    mux_in = n;
            SEL_S:    mux_in = so;
            SEL_E:    mux_in = e;
            SEL_W:    mux_in = w;
            SEL_SELF: mux_in = sf;
            default:  mux_in = 0;
        endcase
    endfunction

    wire [DATA_W-1:0] opa = mux_in(sel_a, in_north, in_south, in_east, in_west,
                                   in_self[DATA_W-1:0]);
    wire [DATA_W-1:0] opb = mux_in(sel_b, in_north, in_south, in_east, in_west,
                                   in_self[DATA_W-1:0]);

    // ------------- unified signed/unsigned multiplier -------------
    // ONE 17x17 signed multiplier per PE.  The operand source and signed
    // interpretation are selected by the opcode, so Vivado packs exactly
    // one DSP48E1 per PE -- not three as in the original split version.
    //
    //   op == OP_MACB_S : operands sign-extended (true signed mul).
    //   op == OP_MACB   : operands zero-extended (unsigned mul, same DSP).
    //   op == OP_MUL    : mesh operands, zero-extended.
    //   op == OP_MAC    : mesh operands, zero-extended + accumulate.
    //
    // Sign-extension adds a single bit at MSB; zero-extension for unsigned
    // keeps the top bit at 0 so the signed multiplier still produces the
    // correct 32-bit unsigned product in its lower bits.  Bit-exact with
    // the old 3-multiplier implementation.
    //
    // PIPELINE=1 inserts AREG/BREG (input) and PREG (output) registers
    // so the whole thing packs into one DSP48E1; Fmax > 200 MHz on Artix-7.

    wire       use_macb    = (op == OP_MACB) | (op == OP_MACB_S);
    wire       signed_mode = (op == OP_MACB_S);

    // pick the operand source based on opcode.  Broadcast (bypass/b_col)
    // for MACB/MACB_S; mesh (opa/opb) for MUL/MAC.  This mux is one LUT
    // level; Vivado sinks it into the DSP input path via AREG/BREG.
    wire [DATA_W-1:0] mul_a_raw = use_macb ? in_bypass : opa;
    wire [DATA_W-1:0] mul_b_raw = use_macb ? in_b_col  : opb;

    // sign/zero extension to (DATA_W+1) = 17 bits
    wire sext_a = signed_mode & mul_a_raw[DATA_W-1];
    wire sext_b = signed_mode & mul_b_raw[DATA_W-1];
    wire signed [DATA_W:0] mul_a_ext = {sext_a, mul_a_raw};
    wire signed [DATA_W:0] mul_b_ext = {sext_b, mul_b_raw};

    // input pipeline register (DSP48E1 AREG/BREG).  No reset: the DSP's
    // synchronous-reset input cannot be driven from an async rst_n, and a
    // fabric FDCE would block DSP packing.  One cycle of power-on garbage
    // is absorbed by clr_acc during the FSM's S_CFG state.
    (* use_dsp = "yes" *) reg signed [DATA_W:0] mul_a_q, mul_b_q;
    always @(posedge clk) begin
        mul_a_q <= mul_a_ext;
        mul_b_q <= mul_b_ext;
    end

    wire signed [DATA_W:0] mul_a_eff_pre = PIPELINE ? mul_a_q : mul_a_ext;
    wire signed [DATA_W:0] mul_b_eff_pre = PIPELINE ? mul_b_q : mul_b_ext;

    // the actual 17x17 -> 34b signed multiply
    wire signed [2*DATA_W+1:0] mul_prod_d = mul_a_eff_pre * mul_b_eff_pre;

    // output pipeline register (DSP48E1 PREG)
    (* use_dsp = "yes" *) reg signed [2*DATA_W+1:0] mul_prod_q;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) mul_prod_q <= 0;
        else        mul_prod_q <= mul_prod_d;

    wire signed [2*DATA_W+1:0] mul_prod_eff = PIPELINE ? mul_prod_q : mul_prod_d;

    // bit-exact aliases for the ALU case items below: the lower ACC_W bits
    // of the 34-bit signed product are the correct 32-bit result for
    // unsigned MUL/MAC/MACB (because unsigned operands have MSB=0 of the
    // 17-bit extension, so the product is positive and < 2^32).
    wire [ACC_W-1:0]         mul_prod_unsigned = mul_prod_eff[ACC_W-1:0];

    // ------------- ALU (ACC_W wide) -------------
    reg [ACC_W-1:0] alu_out;

    // zero-extension helper (everything is unsigned for now — signedness is
    // a future knob, see README §5 future-work).
    localparam EXT_W = ACC_W - DATA_W;

    always @(*) begin
        case(op)
            OP_ADD:    alu_out = { {EXT_W{1'b0}}, (opa + opb) };
            OP_SUB:    alu_out = { {EXT_W{1'b0}}, (opa - opb) };
            OP_MUL:    alu_out = mul_prod_unsigned;          // zero-ext unsigned product
            OP_AND:    alu_out = { {EXT_W{1'b0}}, (opa & opb) };
            OP_OR:     alu_out = { {EXT_W{1'b0}}, (opa | opb) };
            OP_XOR:    alu_out = { {EXT_W{1'b0}}, (opa ^ opb) };
            OP_PASS:   alu_out = { {EXT_W{1'b0}}, opa };
            OP_MUX:    alu_out = { {EXT_W{1'b0}}, (opa[0] ? opb : opa) };
            OP_MAC:    alu_out = mul_prod_unsigned + in_self;          // unsigned MAC
            OP_MACB:   alu_out = mul_prod_unsigned + in_self;          // unsigned broadcast MAC
            OP_MACB_S: alu_out = $signed(mul_prod_eff)
                               + $signed(in_self);                     // signed broadcast MAC
            default:   alu_out = 0;
        endcase
    end

    // ------------- accumulator register -------------
    always @(posedge clk or negedge rst_n)
        if(!rst_n)        out <= 0;
        else if(clr_acc)  out <= 0;
        else if(out_en)   out <= alu_out;

    assign out_mesh = out[DATA_W-1:0];
endmodule
