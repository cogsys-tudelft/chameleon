module flog2 #(
    parameter int BIT_WIDTH = 8
) (
    input  [BIT_WIDTH-1:0] in,
    output reg [$clog2(BIT_WIDTH)-1:0] out
);
    integer i;

    always_comb begin
        out = 0;
        // Skip checking the last bit / LSB (i > 0) instead of (i >= 0)
        // because it doesnt matter whether its zero or not, it will always
        // result in log2 being zero
        for (i = BIT_WIDTH-1; i > 0; i = i - 1) begin
            if (in[i] == 1'b1) begin
                out = i;
                break;
            end
        end
    end

endmodule
