module pe_array_adder_tree #(
    parameter integer PE_OUT_BIT_WIDTH = 4,
    parameter integer ADDER_TREE_OUT_BIT_WIDTH = 24,
    parameter integer SUBSECTION_SUM_BIT_WIDTH = 8,
    parameter integer COLS = 16,
    parameter integer ROWS = 16,
    parameter integer SUBSECTION_SIZE = 4
)(
    input signed [PE_OUT_BIT_WIDTH-1:0] pe_array_out [COLS][ROWS],
    output reg signed [ADDER_TREE_OUT_BIT_WIDTH-1:0] summed_cols [COLS],
    output reg signed [SUBSECTION_SUM_BIT_WIDTH-1:0] summed_subsection_cols [COLS]
);

     // Generate the adder trees
    for (genvar col = 0; col < COLS; col = col + 1) begin: gen_accumulators
        always_comb begin
            summed_subsection_cols[col] = 0;

            for (int sub_row = 0; sub_row < SUBSECTION_SIZE; sub_row = sub_row + 1) begin
                summed_subsection_cols[col] += $signed(pe_array_out[col][sub_row]);
            end

            summed_cols[col] = $signed(summed_subsection_cols[col]);

            for (int m = SUBSECTION_SIZE; m < ROWS; m = m + 1) begin
                summed_cols[col] += $signed(pe_array_out[col][m]);
            end
        end
    end

endmodule
