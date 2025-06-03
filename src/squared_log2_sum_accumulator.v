module squared_log2_sum_accumulator #(
    parameter int COLS = 16,
    parameter int SUB_COLS = 4,
    parameter int BIT_WIDTH = 8,
    parameter int LEFT_SHIFT_FEW_SHOT_SCALE_WIDTH = 2,
    parameter int RIGHT_SHIFT_FEW_SHOT_SCALE_WIDTH = 2,
    parameter int K_SHOT_SCALE_WIDTH = 4,
    parameter int ACCUMULATION_BIT_WIDTH = 16,
    parameter int BIAS_BIT_WIDTH = 14
) (
    input clk,
    input rst,

    input enable,

    input in_4x4_mode,

    input shift_few_shot_data_right,
    input [LEFT_SHIFT_FEW_SHOT_SCALE_WIDTH-1:0] left_shift_few_shot_scale,
    input [RIGHT_SHIFT_FEW_SHOT_SCALE_WIDTH-1:0] right_shift_few_shot_scale,
    input [K_SHOT_SCALE_WIDTH-1:0] k_shot_division_scale,

    input  [$clog2(BIT_WIDTH)-1:0] in [COLS],
    output signed [BIAS_BIT_WIDTH-1:0] out
);
    localparam int SumBitWidth = $clog2((2**(BIT_WIDTH-1))**2*COLS);
    localparam int SubSumBitWidth = $clog2((2**(BIT_WIDTH-1))**2*SUB_COLS);

    reg [SumBitWidth-1:0] col_sum;
    reg [SubSumBitWidth-1:0] sub_col_sum;
    reg [ACCUMULATION_BIT_WIDTH-1:0] accumulator;

    always_comb begin
        sub_col_sum = 0;

        for (int col = 0; col < SUB_COLS; col = col + 1) begin: gen_squared_adder_sub_cols
            sub_col_sum += 4**in[col];
        end

        col_sum = sub_col_sum;

        for (int col = SUB_COLS; col < COLS; col = col + 1) begin: gen_squared_adder_cols
            col_sum += 4**in[col];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            accumulator <= 0;
        end else if (enable) begin
            accumulator <= accumulator + (in_4x4_mode ? sub_col_sum : col_sum);
        end
    end

    assign out = -((shift_few_shot_data_right ? accumulator << right_shift_few_shot_scale : accumulator >> left_shift_few_shot_scale) >> k_shot_division_scale) / 2;

endmodule
