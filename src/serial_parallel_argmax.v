module serial_parallel_argmax
    #(parameter int WIDTH = 8,
      parameter int N = 8,
      parameter int SERIAL_ARGMAX_WIDTH = 4,
      parameter int INPUT_DATA_SHIFT = 0,
      parameter int LEFT_SHIFT_WIDTH = 2,
      localparam int ParallelArgmaxWidth = $clog2(N)
    )(
        input clk,
        input rst,
        input enable,

        input [LEFT_SHIFT_WIDTH-1:0] inputs_left_shift,

        input signed [WIDTH-1:0] data [N],

        output reg [SERIAL_ARGMAX_WIDTH-1:0] argmax,
        output reg signed [WIDTH-1:0] max
    );

    reg [SERIAL_ARGMAX_WIDTH-ParallelArgmaxWidth-1:0] input_counter;
    wire [ParallelArgmaxWidth-1:0] in_argmax;
    wire signed [WIDTH-1:0] in;

    wire signed [WIDTH-INPUT_DATA_SHIFT-1:0] data_shifted [N];

    for (genvar i = 0; i < N; i = i + 1) begin : gen_data_shif
        assign data_shifted[i] = (data[i] <<< inputs_left_shift) >>> INPUT_DATA_SHIFT;
    end

    argmax_tree #(
        .WIDTH(WIDTH-INPUT_DATA_SHIFT),
        .N(N)
    ) argmax_tree_inst (
        .data(data_shifted),
        .argmax(in_argmax),
        .max(in)
    );

    always @(posedge clk) begin
        if (rst) begin
            max <= -2**(WIDTH-1);
            input_counter <= 0;

            argmax <= 0;
        end else if (enable) begin
            if (in > max) begin
                max <= in;

                argmax <= in_argmax + (input_counter * N);
            end

            input_counter <= input_counter + 1;
        end
    end

endmodule
