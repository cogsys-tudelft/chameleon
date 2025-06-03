module managed_bias_memory #(
    parameter BIAS_WORD_BIT_WIDTH = 64,
    parameter BIAS_ROWS = 32,
    parameter START_ADDRESS_BIT_WIDTH = 14,
    parameter MESSAGE_BIT_WIDTH = 32,
    localparam BIAS_ADDRESS_WIDTH = $clog2(BIAS_ROWS)
) (
    input clk,

    input write_new,
    input read_sync,
    input code_is_bias,

    input power_down_small_bias,

    input [START_ADDRESS_BIT_WIDTH-1:0] spi_address,
    input [MESSAGE_BIT_WIDTH-1:0] bias_spi_data_in,
    output [MESSAGE_BIT_WIDTH-1:0] bias_spi_data_out,

    input bias_control_chip_select,
    input bias_control_write_enable,
    input global_power_down,
    input [BIAS_ADDRESS_WIDTH-1:0] bias_control_address,
    input [BIAS_WORD_BIT_WIDTH-1:0] bias_control_data_in,
    input [BIAS_WORD_BIT_WIDTH-1:0] bias_control_mask,

    output [BIAS_WORD_BIT_WIDTH-1:0] bias_data_out
);

    wire bias_chip_select;
    wire bias_write_enable;
    wire [BIAS_ADDRESS_WIDTH-1:0] bias_address;
    wire [BIAS_WORD_BIT_WIDTH-1:0] bias_data_in;
    wire [BIAS_WORD_BIT_WIDTH-1:0] bias_mask;

    memory_manager #(
        .WORD_BIT_WIDTH(BIAS_WORD_BIT_WIDTH),
        .ADDRESS_BIT_WIDTH(BIAS_ADDRESS_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) bias_memory_manager (
        .program_memory_new(write_new),
        .read_memory_sync(read_sync),
        .is_code_for_this_memory(code_is_bias),

        .spi_address (spi_address),
        .spi_data_in (bias_spi_data_in),
        .spi_data_out(bias_spi_data_out),

        .memory_data_out(bias_data_out),

        .control_chip_select(bias_control_chip_select),
        .control_write_enable(bias_control_write_enable),
        .global_power_down(global_power_down),
        .control_address(bias_control_address),
        .control_data_in(bias_control_data_in),
        .control_mask(bias_control_mask),

        .chip_select(bias_chip_select),
        .write_enable(bias_write_enable),
        .read_enable(),
        .program_this_memory_new(),
        .address(bias_address),
        .data_in(bias_data_in),
        .mask(bias_mask)
    );

    bias_memory_wrapper #(
        .WIDTH(BIAS_WORD_BIT_WIDTH),
        .NUM_ROWS(BIAS_ROWS)
    ) bias_memory_wrapper_inst (
        .clk(clk),

        .address(bias_address),
        .data_in(bias_data_in),
        .mask(bias_mask),
        .chip_select(bias_chip_select),
        .write_enable(bias_write_enable),
        .power_down(global_power_down),

        .power_down_small_bias(power_down_small_bias),

        .data_out(bias_data_out)
    );

endmodule
