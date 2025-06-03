module take_lsbs_of_2d_array
#(
    parameter int IN_BIT_WIDTH = 4,
    parameter int OUT_BIT_WIDTH = 4,
    parameter int SCALE_BIT_WIDTH = 4,
    parameter int COLS = 8
) (
    input [IN_BIT_WIDTH-1:0] in[COLS],
    input [SCALE_BIT_WIDTH-1:0] scale,
    output [OUT_BIT_WIDTH-1:0] out[COLS]
);

    for (genvar i = 0; i < COLS; i = i + 1) begin: gen_take_lsbs
        assign out[i] = in[i] >> scale;
    end

endmodule
