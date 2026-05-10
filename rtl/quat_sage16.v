// =============================================================================
// quat_sage16.v  — batched Hamilton quaternion multiply on the 4x4 sage16 fabric
//
// Architecture:
//   * Reuses `matmul_sage16` *verbatim* with OPCODE=OP_MACB_S (signed MAC).
//     The fabric itself is untouched; we only flip one ISA bit.
//   * Builds the 4x4 Q-matrix from q1 combinationally:
//       [ w  -x  -y  -z ]
//       [ x   w  -z   y ]
//       [ y   z   w  -x ]
//       [ z  -y   x   w ]
//     and packs 4 independent q2 vectors as columns of B, so a single
//     matmul produces all four q1*q2_j products at once (16 MACs).
//
// Cycle count: identical to matmul_sage16 (11 cycles, PIPELINE=1).
// =============================================================================
module quat_sage16 #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter CFG_W    = 10,
    parameter PIPELINE = 1
)(
    input                         clk, rst_n,
    input                         start,
    // q1 = [w, x, y, z], signed 16-bit
    input  signed [DATA_W-1:0]    q1_w, q1_x, q1_y, q1_z,
    // q2_flat packs four signed q2 vectors row-major (4*4 = 16 words)
    //   [j*4+0]=w, [j*4+1]=x, [j*4+2]=y, [j*4+3]=z  for j=0..3
    input  [16*DATA_W-1:0]        q2_flat,
    // c_out layout: c_out[j*4+k] = k-th component of q1*q2_j, signed 32-bit
    output [16*ACC_W-1:0]         c_out,
    output                        done
);
    localparam [3:0] OP_MACB_S = 4'd10;

    // ---- build A = Q-matrix of q1 (signed), row-major ----
    wire signed [DATA_W-1:0] A [0:3][0:3];
    assign A[0][0] =  q1_w;  assign A[0][1] = -q1_x;
    assign A[0][2] = -q1_y;  assign A[0][3] = -q1_z;
    assign A[1][0] =  q1_x;  assign A[1][1] =  q1_w;
    assign A[1][2] = -q1_z;  assign A[1][3] =  q1_y;
    assign A[2][0] =  q1_y;  assign A[2][1] =  q1_z;
    assign A[2][2] =  q1_w;  assign A[2][3] = -q1_x;
    assign A[3][0] =  q1_z;  assign A[3][1] = -q1_y;
    assign A[3][2] =  q1_x;  assign A[3][3] =  q1_w;

    // ---- unpack q2 as signed words: q2[j][k] = k-th comp of q2_j ----
    wire signed [DATA_W-1:0] q2 [0:3][0:3];
    genvar gj, gi;
    generate for(gj=0; gj<4; gj=gj+1) begin : uq
        for(gi=0; gi<4; gi=gi+1) begin : ui
            assign q2[gj][gi] = q2_flat[(gj*4+gi)*DATA_W +: DATA_W];
        end
    end endgenerate

    // ---- pack flat a_in (A, row-major) and b_in (columns = q2_j) ----
    // matmul computes C[i][j] = sum_k A[i][k] * B[k][j].
    // To get C[i][j] = (q1 * q2_j)[i], we need B[k][j] = q2_j[k].
    wire [16*DATA_W-1:0] a_flat;
    wire [16*DATA_W-1:0] b_flat;
    generate for(gi=0; gi<4; gi=gi+1) begin : pkA
        for(gj=0; gj<4; gj=gj+1) begin : pkAj
            assign a_flat[(gi*4+gj)*DATA_W +: DATA_W] = A[gi][gj];
            assign b_flat[(gi*4+gj)*DATA_W +: DATA_W] = q2[gj][gi];  // B[k=gi][j=gj] = q2_gj[gi]
        end
    end endgenerate

    // ---- hand off to matmul_sage16 with signed opcode ----
    wire [16*ACC_W-1:0] c_mat;
    matmul_sage16 #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .CFG_W   (CFG_W),
        .PIPELINE(PIPELINE),
        .OPCODE  (OP_MACB_S)
    ) u_mm (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .a_in (a_flat),
        .b_in (b_flat),
        .c_out(c_mat),
        .done (done)
    );

    // ---- transpose to user layout: c_out[j*4+k] = C[k][j] ----
    genvar oi, oj;
    generate for(oi=0; oi<4; oi=oi+1) begin : tI
        for(oj=0; oj<4; oj=oj+1) begin : tJ
            // C[oi][oj] lives at matmul index oi*4+oj; user wants it at oj*4+oi
            assign c_out[(oj*4+oi)*ACC_W +: ACC_W] =
                   c_mat [(oi*4+oj)*ACC_W +: ACC_W];
        end
    end endgenerate
endmodule
