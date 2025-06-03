module activation_memory_wrapper #(
    parameter WIDTH = 1024,
    parameter NUM_ROWS = 128,
    localparam ADDRESS_WIDTH = $clog2(NUM_ROWS)
) (
    input clk,

    input [ADDRESS_WIDTH-1:0] address_write,
    input [ADDRESS_WIDTH-1:0] address_read,
    input [WIDTH-1:0] data_in,
    input [WIDTH-1:0] mask,
    input read_enable,
    input write_enable,
    input power_down,

    output [WIDTH-1:0] data_out
);

`ifndef SYNTHESIZE_FOR_SILICON
    double_port_type_t_sram #(
        .WIDTH(WIDTH),
        .NUM_ROWS(NUM_ROWS)
    ) activation_memory_inst (
        .CLK(clk),

        .REB(~read_enable),
        .WEB(~write_enable),
        .AA (address_write),
        .AB (address_read),

        .D(data_in),
        .M(~mask),
        .Q(data_out)
    );
`else
    real_activation_memory activation_memory_inst (
        .CLK(clk),

        .AA(address_write),
        .AB(address_read),
        .D (data_in),
        .M (mask),

        .PD(power_down),

        .REB(read_enable),
        .WEB(write_enable),

        .Q(data_out)
    );
`endif

endmodule
