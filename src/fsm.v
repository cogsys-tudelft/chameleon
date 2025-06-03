`include "states.vh"

module fsm #(
    localparam STATE_BIT_WIDTH = $clog2(`NUMBER_OF_STATES)
) (
    input clk,
    input rst,

    input continuous_processing,
    input classification,
    input is_output_layer,

    input toggle_processing_new,
    input start_sending,
    input done_sending,

    input skip_sending,
    input start_few_shot_processing,
    input done_few_shot_processing,

    output reg [STATE_BIT_WIDTH-1:0] state,
    output reg [STATE_BIT_WIDTH-1:0] next_state
);

    always @(posedge clk) begin
        if (rst) state <= `IDLE;
        else state <= next_state;
    end

    // Next state logic
    always_comb begin
        case (state)
            `IDLE:
            if (toggle_processing_new) next_state = `RUNNING;
            else next_state = `IDLE;
            `RUNNING:
            if (start_sending) next_state = `SENDING;
            else if (start_few_shot_processing) next_state = `PROCESSING_FEW_SHOT;
            else if (toggle_processing_new | skip_sending) next_state = `IDLE;
            else next_state = `RUNNING;
            `SENDING:
            if (done_sending)
                if (continuous_processing | (~classification & is_output_layer)) next_state = `RUNNING;
                else next_state = `IDLE;
            else next_state = `SENDING;
            `PROCESSING_FEW_SHOT:  // TODO!
            if (done_few_shot_processing) next_state = `IDLE;
            else next_state = `PROCESSING_FEW_SHOT;
            default: next_state = `IDLE;
        endcase
    end

endmodule
