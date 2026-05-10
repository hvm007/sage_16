// =============================================================================
// sage16_pcpi.v
//
// PicoRV32 PCPI (Pico Co-Processor Interface) wrapper around matmul_sage16.
// Exposes the 4x4 sage16 matmul as 4 custom RISC-V instructions.
//
// Custom instructions (R-type, opcode=0b0001011 "custom-0", funct7=0):
//
//   | mnemonic   | funct3 | rs1        | rs2        | rd  | effect                         |
//   |------------|--------|------------|------------|-----|--------------------------------|
//   | sage16.wrA   | 000    | idx (4b)   | val (16b)  |  -  | a_buf[idx] <= val              |
//   | sage16.wrB   | 001    | idx (4b)   | val (16b)  |  -  | b_buf[idx] <= val              |
//   | sage16.go    | 010    |  -         |  -         |  -  | start matmul; blocks until done|
//   | sage16.rdC   | 011    | idx (4b)   |  -         |  rd | rd = {16'd0, c_buf[idx]}       |
//
// Inline-asm examples (GCC):
//   #define sage16_WRA(idx,val) asm volatile (".insn r 0x0B,0,0,x0,%0,%1"::"r"(idx),"r"(val))
//   #define sage16_WRB(idx,val) asm volatile (".insn r 0x0B,1,0,x0,%0,%1"::"r"(idx),"r"(val))
//   #define sage16_GO()         asm volatile (".insn r 0x0B,2,0,x0,x0,x0")
//   static inline int sage16_RDC(int idx) {
//       int r;
//       asm volatile (".insn r 0x0B,3,0,%0,%1,x0" : "=r"(r) : "r"(idx));
//       return r;
//   }
//
// PCPI protocol:
//   - Core asserts pcpi_valid when it decodes an unimplemented instruction.
//   - If insn is ours, we assert pcpi_wait while processing, then pulse
//     pcpi_ready for 1 cycle. If returning a value, pcpi_wr=1 and pcpi_rd=value
//     in the same cycle as pcpi_ready.
//   - If insn is NOT ours, we keep all outputs low so the core can trap
//     (or another coprocessor can claim it).
// =============================================================================
module sage16_pcpi #(
    parameter DATA_W   = 16,
    parameter ACC_W    = 32,
    parameter PIPELINE = 1
)(
    input                 clk, rst_n,

    // --- PCPI slave port (connects to PicoRV32) ---
    input                 pcpi_valid,
    input      [31:0]     pcpi_insn,
    input      [31:0]     pcpi_rs1,
    input      [31:0]     pcpi_rs2,
    output reg            pcpi_wr,
    output reg [31:0]     pcpi_rd,
    output reg            pcpi_wait,
    output reg            pcpi_ready
);
    // --------------------------------------------------------------
    // Instruction decode
    // --------------------------------------------------------------
    localparam [6:0] sage16_OPCODE = 7'b0001011;  // RISC-V "custom-0"

    localparam [2:0] F_WRA = 3'd0,
                     F_WRB = 3'd1,
                     F_GO  = 3'd2,
                     F_RDC = 3'd3;

    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] funct3 = pcpi_insn[14:12];
    wire [6:0] funct7 = pcpi_insn[31:25];
    wire       is_ours = pcpi_valid &&
                         (opcode == sage16_OPCODE) &&
                         (funct7 == 7'd0);

    wire [3:0]        idx = pcpi_rs1[3:0];
    wire [DATA_W-1:0] val = pcpi_rs2[DATA_W-1:0];

    // --------------------------------------------------------------
    // A and B buffers (16 words each, 16-bit)
    // --------------------------------------------------------------
    reg [DATA_W-1:0] a_buf [0:15];
    reg [DATA_W-1:0] b_buf [0:15];

    wire [16*DATA_W-1:0] a_flat, b_flat;
    wire [16*ACC_W-1:0]  c_flat;
    genvar g;
    generate for(g=0; g<16; g=g+1) begin : gbuf
        assign a_flat[g*DATA_W +: DATA_W] = a_buf[g];
        assign b_flat[g*DATA_W +: DATA_W] = b_buf[g];
    end endgenerate

    // --------------------------------------------------------------
    // sage16 accelerator instance
    // --------------------------------------------------------------
    reg  sage16_start;
    wire sage16_done;

    matmul_sage16 #(
        .DATA_W  (DATA_W),
        .ACC_W   (ACC_W),
        .PIPELINE(PIPELINE)
    ) u_mm(
        .clk(clk), .rst_n(rst_n),
        .start(sage16_start),
        .a_in(a_flat), .b_in(b_flat),
        .c_out(c_flat), .done(sage16_done)
    );

    // --------------------------------------------------------------
    // PCPI FSM for sage16.go (multi-cycle; wrA/wrB/rdC are 1-cycle)
    // --------------------------------------------------------------
    localparam [1:0] GO_IDLE = 2'd0,
                     GO_FIRE = 2'd1,
                     GO_WAIT = 2'd2;
    reg [1:0] go_state;

    integer ii;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            pcpi_wr    <= 0;
            pcpi_rd    <= 32'd0;
            pcpi_wait  <= 0;
            pcpi_ready <= 0;
            sage16_start <= 0;
            go_state   <= GO_IDLE;
            for(ii=0; ii<16; ii=ii+1) begin
                a_buf[ii] <= 0;
                b_buf[ii] <= 0;
            end
        end else begin
            // default pulse-low
            pcpi_wr    <= 0;
            pcpi_ready <= 0;
            pcpi_wait  <= 0;
            sage16_start <= 0;

            if(is_ours) begin
                case(funct3)
                // ----- sage16.wrA : a_buf[idx] <= val, 1 cycle -----
                F_WRA: begin
                    a_buf[idx] <= val;
                    pcpi_ready <= 1;
                end
                // ----- sage16.wrB : b_buf[idx] <= val, 1 cycle -----
                F_WRB: begin
                    b_buf[idx] <= val;
                    pcpi_ready <= 1;
                end
                // ----- sage16.go : pulse start, stall core, wait done -----
                F_GO: begin
                    case(go_state)
                    GO_IDLE: begin
                        sage16_start <= 1;
                        pcpi_wait  <= 1;
                        go_state   <= GO_FIRE;
                    end
                    GO_FIRE: begin
                        pcpi_wait <= 1;          // FSM is running
                        go_state  <= GO_WAIT;
                    end
                    GO_WAIT: begin
                        if(sage16_done) begin
                            pcpi_wait  <= 0;
                            pcpi_ready <= 1;     // 1-cycle ack
                            go_state   <= GO_IDLE;
                        end else begin
                            pcpi_wait <= 1;
                        end
                    end
                    default: go_state <= GO_IDLE;
                    endcase
                end
                // ----- sage16.rdC : rd <= c[idx] (full ACC_W, zero-extended to 32b) -----
                F_RDC: begin
                    pcpi_rd    <= c_flat[idx*ACC_W +: ACC_W];
                    pcpi_wr    <= 1;
                    pcpi_ready <= 1;
                end
                default: ;
                endcase
            end else begin
                // Not our insn OR pcpi_valid went low -> reset go_state
                // (e.g. core trapped or moved on)
                if(!pcpi_valid) go_state <= GO_IDLE;
            end
        end
    end
endmodule
