module bias_memory_wrapper #(
    parameter WIDTH = 64,
    parameter NUM_ROWS = 32,
    localparam ADDRESS_WIDTH = $clog2(NUM_ROWS)
) (
    input clk,

    input [ADDRESS_WIDTH-1:0] address,
    input [WIDTH-1:0] data_in,
    input [WIDTH-1:0] mask,
    input chip_select,
    input write_enable,
    input power_down,

    input power_down_small_bias,

    output [WIDTH-1:0] data_out
);

`ifndef SYNTHESIZE_FOR_SILICON
    single_port_type_t_sram #(
        .WIDTH(WIDTH),
        .NUM_ROWS(NUM_ROWS)
    ) bias_memory_inst (
        .CLK(clk),

        .A(address),
        .D(data_in),
        .M(~mask),

        .CEB(~chip_select),
        .WEB(~write_enable),

        .Q(data_out)
    );
`else
    real_bias_memory bias_memory_inst (
        .CLK(clk),

        .A(address),
        .D(data_in),
        .M(mask),

        .PD(power_down),

        .CEB(chip_select),
        .WEB(write_enable),

        .Q(data_out),

        .PD_B1(power_down_small_bias)
    );
`endif

endmodule
