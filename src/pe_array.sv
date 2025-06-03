module pe_array #(
    parameter integer ROWS = 16,
    parameter integer COLS = 16,
    parameter integer ACTIVATION_BIT_WIDTH = 4,
    parameter integer WEIGHT_BIT_WIDTH = 4,
    parameter integer BIAS_BIT_WIDTH = 20,
    parameter integer SCALE_BIT_WIDTH = 4,
    parameter integer ACCUMULATION_BIT_WIDTH = 24,
    parameter integer SUBSECTION_SIZE = 4
) (
    input clk,

    input enable,

    input apply_identity,
    input apply_in_scale,
    input use_subsection,
    input apply_bias,

    input [ACTIVATION_BIT_WIDTH-1:0] in [ROWS],
`ifdef FLAT_WEIGHTS
    input [WEIGHT_BIT_WIDTH-1:0] weights [ROWS*COLS],
`else
    input [WEIGHT_BIT_WIDTH-1:0] weights [ROWS][COLS],
`endif
    input signed [BIAS_BIT_WIDTH-1:0] biases [COLS],
    input [SCALE_BIT_WIDTH-1:0] out_scale,
    input [SCALE_BIT_WIDTH-1:0] in_scale,

    output reg [ACTIVATION_BIT_WIDTH-1:0] out [COLS],

    output signed [ACCUMULATION_BIT_WIDTH-1:0] col_accumulator [COLS]
);
    localparam integer PeOutBitWidth = ACTIVATION_BIT_WIDTH + (2 ** WEIGHT_BIT_WIDTH) / 2;
    localparam integer AdderTreeOutBitWidth = PeOutBitWidth + $clog2(ROWS);
    localparam integer DefaultShiftShift = 2 ** (WEIGHT_BIT_WIDTH - 1) - 1;
    localparam integer AccumulationBitWidthMinusOne = ACCUMULATION_BIT_WIDTH - 1;
    localparam integer SubsectionSumBitWidth = PeOutBitWidth + $clog2(SUBSECTION_SIZE);

    if (AdderTreeOutBitWidth > ACCUMULATION_BIT_WIDTH) begin : gen_raise_error__AdderTreeOutBitWidth_must_be_less_than_or_equal_to_ACCUMULATION_BIT_WIDTH
        $fatal(1, "ERROR: AdderTreeOutBitWidth must be less than or equal to ACCUMULATION_BIT_WIDTH");
    end

    // Array order is switched to make it easier to do full column
    // access (as this is now the first dimension)
    wire signed [PeOutBitWidth-1:0] pe_array_out [COLS][ROWS];
    reg signed [AdderTreeOutBitWidth-1:0] summed_cols [COLS];
    reg signed [SubsectionSumBitWidth-1:0] summed_subsection_cols [COLS];

    reg [SCALE_BIT_WIDTH-1:0] out_scale_reg;

    always @(posedge clk) begin
        if (enable & apply_bias) begin
            out_scale_reg <= out_scale;
        end
    end

    // TODO: could gate the weights and inputs + biases with enable

    // Create a 2D array of PEs
    for (genvar row = 0; row < ROWS; row = row + 1) begin: gen_rows
        for (genvar col = 0; col < COLS; col = col + 1) begin: gen_cols
            wire [WEIGHT_BIT_WIDTH-1:0] weight;

`ifdef FLAT_WEIGHTS
            assign weight = weights[row * ROWS + col];
`else
            assign weight = weights[row][col];
`endif
            pe #(.WEIGHT_BIT_WIDTH(WEIGHT_BIT_WIDTH), .INPUT_BIT_WIDTH(ACTIVATION_BIT_WIDTH)) pe_inst (
                .in(in[row]),
                .weight(weight),
                .out(pe_array_out[col][row])
            );
        end
    end

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

    // Create a 1D array of output PEs
    for (genvar l = 0; l < COLS; l = l + 1) begin: gen_sum_processors
        reg signed [ACCUMULATION_BIT_WIDTH-1:0] accumulator;
        reg signed [ACCUMULATION_BIT_WIDTH-1:0] pe_array_sum;
        reg signed [ACCUMULATION_BIT_WIDTH-1:0] accumulator_new;

        wire inside_subsection = l < SUBSECTION_SIZE;
        wire outside_active_subsection = use_subsection & ~inside_subsection;

        always_comb begin
            if (apply_identity) begin
                pe_array_sum = in[l];
            end else if (use_subsection & inside_subsection) begin
                pe_array_sum = $signed(summed_subsection_cols[l]);
            end else begin
                pe_array_sum = $signed(summed_cols[l]);
            end

            if (apply_in_scale)
                pe_array_sum = pe_array_sum <<< in_scale;

            accumulator_new = pe_array_sum + (apply_bias == 1'b1 ? $signed(biases[l]) : accumulator);
        end

        always @(posedge clk) begin
            if (enable & (use_subsection ? inside_subsection : 1'b1)) begin
                accumulator <= accumulator_new;
            end
        end

        always_comb begin
            // ReLU implementation
            if (accumulator > 0) begin
                // Set all values outside the subsection, when the subsection is active, to zero
                // TODO: check if this can be deleted for improved PPA
                if (outside_active_subsection) begin
                    out[l] = 0;
                end else begin
                    /* verilator lint_off WIDTHTRUNC */
                    out[l] = accumulator >> out_scale_reg;
                    /* verilator lint_on WIDTHTRUNC */
                end
            end else begin
                out[l] = 0;
            end
        end

        // TO DO: can move this into main chameleon.sv file, because this assignment is not necessary for the bias computation, only for the argmax
        // Store the most negative value in the column accumulator if subsection mode is action and we are outside the active subsection,
        // otherwise just shift the accumulator a little bit so that it is less heavy to compute the argmax later
        /* verilator lint_off WIDTHTRUNC */
        assign col_accumulator[l] = outside_active_subsection ? {{1'b1}, {AccumulationBitWidthMinusOne{1'b0}}} : accumulator;
        /* verilator lint_on WIDTHTRUNC */
    end

endmodule
