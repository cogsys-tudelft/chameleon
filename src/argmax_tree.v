module argmax_tree #(
    parameter int WIDTH = 8,
    parameter int N = 8,
    localparam int ArgmaxWidth = $clog2(N)
)(
    input signed [WIDTH-1:0] data [N],
    output [ArgmaxWidth-1:0] argmax,
    output signed [WIDTH-1:0] max
);

    if (N % 2 != 0) begin
        $fatal(1, "Only even N is supported for now");
    end

    localparam int STAGES = $clog2(N);

    // Intermediate wires
    reg signed [WIDTH-1:0] stage_max[STAGES:0][N];
    reg [ArgmaxWidth-1:0] stage_argmax[STAGES:0][N];

    always_comb begin
        // Initial stage assignment
        for (int i = 0; i < N; i = i + 1) begin : init
            stage_max[0][i] = data[i];
            /* verilator lint_off WIDTHTRUNC */
            stage_argmax[0][i] = i;
            /* verilator lint_on WIDTHTRUNC */
        end

        for (int s = 0; s < STAGES; s = s + 1) begin : stages
            for (int j = 0; j < (N >> (s + 1)); j = j + 1) begin : comparators
                reg signed [WIDTH-1:0] max1, max2;
                reg [ArgmaxWidth-1:0] argmax1, argmax2;

                max1 = stage_max[s][2*j];
                max2 = stage_max[s][2*j+1];
                argmax1 = stage_argmax[s][2*j];
                argmax2 = stage_argmax[s][2*j+1];

                if (max1 >= max2) begin
                    stage_max[s+1][j] = max1;
                    stage_argmax[s+1][j] = argmax1;
                end else begin
                    stage_max[s+1][j] = max2;
                    stage_argmax[s+1][j] = argmax2;
                end
            end

            if ((N >> s) % 2 != 0) begin : odd_case
                stage_max[s+1][(N >> (s + 1))] = stage_max[s][N >> s];
                stage_argmax[s+1][(N >> (s + 1))] = stage_argmax[s][N >> s];
            end
        end
    end

    // Final assignment
    assign max = stage_max[STAGES][0];
    assign argmax = stage_argmax[STAGES][0];

endmodule
