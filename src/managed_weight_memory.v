module managed_weight_memory #(
    parameter WEIGHT_WORD_BIT_WIDTH = 64,
    parameter WEIGHT_ROWS = 32,
    parameter START_ADDRESS_BIT_WIDTH = 14,
    parameter MESSAGE_BIT_WIDTH = 32,
    localparam WEIGHT_ADDRESS_WIDTH = $clog2(WEIGHT_ROWS)
) (
    input clk,
    input rst,

    input write_new,
    input read_sync,
    input code_is_weight,

    input in_4x4_mode,

    input [START_ADDRESS_BIT_WIDTH-1:0] spi_address,
    input [MESSAGE_BIT_WIDTH-1:0] weights_spi_data_in,
    output [MESSAGE_BIT_WIDTH-1:0] weight_spi_data_out,

    input weight_control_chip_select,
    input weight_control_write_enable,
    input global_power_down,
    input [WEIGHT_ADDRESS_WIDTH-1:0] weight_control_address,
    input [WEIGHT_WORD_BIT_WIDTH-1:0] weight_control_data_in,
    input [WEIGHT_WORD_BIT_WIDTH-1:0] weight_control_mask,

    output [WEIGHT_WORD_BIT_WIDTH-1:0] weight_data_out
);

    wire weight_chip_select;
    wire weight_write_enable;
    wire [WEIGHT_ADDRESS_WIDTH-1:0] weight_address;
    wire [WEIGHT_WORD_BIT_WIDTH-1:0] weight_data_in;
    wire [WEIGHT_WORD_BIT_WIDTH-1:0] weight_mask;

    memory_manager #(
        .WORD_BIT_WIDTH(WEIGHT_WORD_BIT_WIDTH),
        .ADDRESS_BIT_WIDTH(WEIGHT_ADDRESS_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) weight_memory_manager (
        .program_memory_new(write_new),
        .read_memory_sync(read_sync),
        .is_code_for_this_memory(code_is_weight),

        .spi_address (spi_address),
        .spi_data_in (weights_spi_data_in),
        .spi_data_out(weight_spi_data_out),

        .memory_data_out(weight_data_out),

        .control_chip_select(weight_control_chip_select),
        .control_write_enable(weight_control_write_enable),
        .global_power_down(global_power_down),
        .control_address(weight_control_address),
        .control_data_in(weight_control_data_in),
        .control_mask(weight_control_mask),

        .chip_select(weight_chip_select),
        .write_enable(weight_write_enable),
        .read_enable(),
        .program_this_memory_new(),
        .address(weight_address),
        .data_in(weight_data_in),
        .mask(weight_mask)
    );

    weight_memory_wrapper #(
        .WIDTH(WEIGHT_WORD_BIT_WIDTH),
        .NUM_ROWS(WEIGHT_ROWS)
    ) weight_memory_wrapper_inst (
        .clk(clk),
        .rst(rst),

        .address(weight_address),
        .data_in(weight_data_in),
        .mask(weight_mask),
        .chip_select(weight_chip_select),
        .write_enable(weight_write_enable),
        .power_down(global_power_down),

        .in_4x4_mode(in_4x4_mode),

        .data_out(weight_data_out)
    );

endmodule
