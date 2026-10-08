/**
 * Wrapper to run the RTL system tests on the post-synthesis netlist of chameleon.
 *
 * The netlist has no parameters anymore, but the tests read them from the DUT, so
 * they are repeated here. They must match the values the netlist was synthesized with
 * (the defaults of chameleon.sv).
 */
module chameleon_netlist_tb #(
    parameter int HIGH_SPEED_IN_PINS  = 16,
    parameter int HIGH_SPEED_OUT_PINS = 8,

    parameter int MESSAGE_BIT_WIDTH = 32,
    parameter int CODE_BIT_WIDTH = 4,
    parameter int START_ADDRESS_BIT_WIDTH = 16,

    parameter int PE_ROWS = 16,
    parameter int PE_COLS = 16,
    parameter int SUBSECTION_SIZE = 4,

    parameter int ACTIVATION_BIT_WIDTH = 4,
    parameter int WEIGHT_BIT_WIDTH = 4,
    parameter int BIAS_BIT_WIDTH = 14,
    parameter int SCALE_BIT_WIDTH = 4,
    parameter int ACCUMULATION_BIT_WIDTH = 18,

    parameter int MAX_NUM_LOGITS = 1024,
    parameter int MAX_SHOTS = 127,
    parameter int FEW_SHOT_ACCUMULATION_BIT_WIDTH = 20,

    parameter int ACTIVATION_ROWS = 256,
    parameter int WEIGHT_ROWS = 1024,
    parameter int BIAS_ROWS = 128,
    parameter int INPUT_ROWS = 32,

    parameter int MAX_NUM_LAYERS = 32,
    parameter int MAX_NUM_CHANNELS = 1024,
    parameter int MAX_KERNEL_SIZE = 15,

    parameter int ARGMAX_INPUTS_SHIFT = 0,

    parameter int WAIT_CYCLES_WIDTH = 4,

    parameter int CLOCK_DIVIDER_STAGES = 7
) (
    input clk_ext,
    input rst_async,
    input enable_clk_int,

    input  toggle_processing,
    input  is_new_task,
    output in_idle,

    input  SCK,
    output MISO,
    input  MOSI,

    output clk_int_div,

    input [HIGH_SPEED_IN_PINS-1:0] data_in,
    input in_request,
    output out_acknowledge,

    output [HIGH_SPEED_OUT_PINS-1:0] data_out,
    output out_request,
    input in_acknowledge
);

    chameleon chameleon_inst (
        .clk_ext(clk_ext),
        .rst_async(rst_async),
        .enable_clk_int(enable_clk_int),

        .toggle_processing(toggle_processing),
        .is_new_task(is_new_task),
        .in_idle(in_idle),

        .SCK (SCK),
        .MISO(MISO),
        .MOSI(MOSI),

        .clk_int_div(clk_int_div),

        .data_in(data_in),
        .in_request(in_request),
        .out_acknowledge(out_acknowledge),

        .data_out(data_out),
        .out_request(out_request),
        .in_acknowledge(in_acknowledge)
    );

endmodule
