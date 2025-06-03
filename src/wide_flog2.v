module wide_flog2 #(
    parameter int COLS = 16,
    parameter int BIT_WIDTH = 8
) (
    input  [BIT_WIDTH-1:0] in [COLS],
    output [$clog2(BIT_WIDTH)-1:0] out [COLS]
);
    for (genvar i = 0; i < COLS; i = i + 1) begin : gen_log2_calc
        flog2 #(
            .BIT_WIDTH(BIT_WIDTH)
        ) log2_inst (
            .in(in[i]),
            .out(out[i])
        );
    end

endmodule
