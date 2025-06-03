module convert_2d_to_1d_array_size
// Layout of 'out' port from right to left:
// column 0, column 1, column 2, ...
#(
    parameter int BIT_WIDTH = 4,
    parameter int OUT_BIT_WIDTH = -1,
    parameter int COLS = 8,
    localparam int FinalOutBitWidth = (OUT_BIT_WIDTH == -1) ? BIT_WIDTH : OUT_BIT_WIDTH
) (
    input [BIT_WIDTH-1:0] in[COLS],
    output [COLS*FinalOutBitWidth-1:0] out
);
    for (genvar i = 0; i < COLS; i = i + 1) begin : gen_convert
        assign out[FinalOutBitWidth+(i*FinalOutBitWidth)-1-:FinalOutBitWidth] = in[i];
    end
endmodule
