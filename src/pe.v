module pe
    // Output bit width is the sum of input bit width and 1 sign bit and the maximum
    // possible, left shift, which is equal to (2^WEIGHT_BIT_WIDTH)/2-1 
    #(
        parameter integer WEIGHT_BIT_WIDTH = 4,
        parameter integer INPUT_BIT_WIDTH = 4,
        localparam integer OutputBitWidth = INPUT_BIT_WIDTH + (2**WEIGHT_BIT_WIDTH)/2
    ) (
        input [INPUT_BIT_WIDTH-1:0] in,
        input [WEIGHT_BIT_WIDTH-1:0] weight,
        output signed [OutputBitWidth-1:0] out
    );

    wire weight_sign;
    wire [WEIGHT_BIT_WIDTH-2:0] weight_abs;

    assign {weight_sign, weight_abs} = weight;

    assign out = (weight_sign == 1'b1 ? -in : in) <<< weight_abs;

endmodule
