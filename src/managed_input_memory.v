module managed_input_memory #(
    parameter INPUT_WORD_BIT_WIDTH = 64,
    parameter INPUT_ROWS = 32,
    parameter START_ADDRESS_BIT_WIDTH = 14,
    parameter MESSAGE_BIT_WIDTH = 32,
    localparam INPUT_ADDRESS_WIDTH = $clog2(INPUT_ROWS)
) (
    input clk,

    input write_new,
    input read_sync,
    input code_is_input,

    input [START_ADDRESS_BIT_WIDTH-1:0] spi_address,
    input [MESSAGE_BIT_WIDTH-1:0] input_spi_data_in,
    output [MESSAGE_BIT_WIDTH-1:0] input_spi_data_out,

    input input_control_read_enable,
    input input_control_write_enable,
    input [INPUT_ADDRESS_WIDTH-1:0] input_control_address_read,
    input [INPUT_ADDRESS_WIDTH-1:0] input_control_address_write,
    input [INPUT_WORD_BIT_WIDTH-1:0] input_control_data_in,
    input [INPUT_WORD_BIT_WIDTH-1:0] input_control_mask,

    output [INPUT_WORD_BIT_WIDTH-1:0] input_data_out
);

    wire input_chip_select;
    wire input_write_enable;
    wire input_read_enable;
    wire program_this_memory_new;
    wire [INPUT_ADDRESS_WIDTH-1:0] input_address;
    wire [INPUT_WORD_BIT_WIDTH-1:0] input_data_in;
    wire [INPUT_WORD_BIT_WIDTH-1:0] input_mask;

    memory_manager #(
        .WORD_BIT_WIDTH(INPUT_WORD_BIT_WIDTH),
        .ADDRESS_BIT_WIDTH(INPUT_ADDRESS_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) input_memory_manager (
        .program_memory_new(write_new),
        .read_memory_sync(read_sync),
        .is_code_for_this_memory(code_is_input),

        .spi_address (spi_address),
        .spi_data_in (input_spi_data_in),
        .spi_data_out(input_spi_data_out),

        .memory_data_out(input_data_out),

        .control_chip_select(input_control_read_enable),
        .control_write_enable(input_control_write_enable),
        .global_power_down(1'b0),
        .control_address(input_control_address_read),
        .control_data_in(input_control_data_in),
        .control_mask(input_control_mask),

        .chip_select(),
        .write_enable(input_write_enable),
        .read_enable(input_read_enable),
        .program_this_memory_new(program_this_memory_new),
        .address(input_address),
        .data_in(input_data_in),
        .mask(input_mask)
    );

    double_port_type_t_sram #(
        .WIDTH(INPUT_WORD_BIT_WIDTH),
        .NUM_ROWS(INPUT_ROWS)
    ) double_port_input_memory (
        .CLK(clk),

        .REB(~input_read_enable),
        .WEB(~input_write_enable),
        .AA (program_this_memory_new ? input_address : input_control_address_write),
        .AB (input_address),

        .D(input_data_in),
        .M(~input_mask),
        .Q(input_data_out)
    );

endmodule
