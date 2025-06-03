module managed_activation_memory #(
    parameter ACTIVATION_WORD_BIT_WIDTH = 64,
    parameter ACTIVATION_ROWS = 32,
    parameter START_ADDRESS_BIT_WIDTH = 14,
    parameter MESSAGE_BIT_WIDTH = 32,
    localparam ACTIVATION_ADDRESS_WIDTH = $clog2(ACTIVATION_ROWS)
) (
    input clk,

    input write_new,
    input read_sync,
    input code_is_activation,

    input [START_ADDRESS_BIT_WIDTH-1:0] spi_address,
    input [MESSAGE_BIT_WIDTH-1:0] activations_spi_data_in,
    output [MESSAGE_BIT_WIDTH-1:0] activation_spi_data_out,

    input activation_control_read_enable,
    input activation_control_write_enable,
    input global_power_down,
    input [ACTIVATION_ADDRESS_WIDTH-1:0] activation_control_address_read,
    input [ACTIVATION_ADDRESS_WIDTH-1:0] activation_control_address_write,
    input [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_control_data_in,
    input [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_control_mask,

    output [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_data_out
);

    wire activation_chip_select;
    wire activation_write_enable;
    wire activation_read_enable;
    wire program_this_memory_new;
    wire [ACTIVATION_ADDRESS_WIDTH-1:0] activation_address;
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_data_in;
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_mask;

    memory_manager #(
        .WORD_BIT_WIDTH(ACTIVATION_WORD_BIT_WIDTH),
        .ADDRESS_BIT_WIDTH(ACTIVATION_ADDRESS_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) activation_memory_manager (
        .program_memory_new(write_new),
        .read_memory_sync(read_sync),
        .is_code_for_this_memory(code_is_activation),

        .spi_address (spi_address),
        .spi_data_in (activations_spi_data_in),
        .spi_data_out(activation_spi_data_out),

        .memory_data_out(activation_data_out),

        .control_chip_select(activation_control_read_enable),
        .control_write_enable(activation_control_write_enable),
        .global_power_down(global_power_down),
        .control_address(activation_control_address_read),
        .control_data_in(activation_control_data_in),
        .control_mask(activation_control_mask),

        .chip_select(),
        .write_enable(activation_write_enable),
        .read_enable(activation_read_enable),
        .program_this_memory_new(program_this_memory_new),
        .address(activation_address),
        .data_in(activation_data_in),
        .mask(activation_mask)
    );

    activation_memory_wrapper #(
        .WIDTH(ACTIVATION_WORD_BIT_WIDTH),
        .NUM_ROWS(ACTIVATION_ROWS)
    ) activation_memory_wrapper_inst (
        .clk(clk),

        .address_write(program_this_memory_new ? activation_address : activation_control_address_write),
        .address_read(activation_address),
        .data_in(activation_data_in),
        .mask(activation_mask),
        .write_enable(activation_write_enable),
        .read_enable(activation_read_enable),
        .power_down(global_power_down),

        .data_out(activation_data_out)
    );

endmodule
