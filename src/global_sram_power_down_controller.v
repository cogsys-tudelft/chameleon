module global_sram_power_down_controller
/**
  * Make sure that the chip select or read and write enable are both low in the cycle
  * before the power down is high. This is required for correct power-down operation
  * according to TxxC spec.
  */
#(
    parameter int WAIT_CYCLES_WIDTH = 10
) (
    input clk,
    input rst,

    input should_be_on,
    input allow_power_down,

    input [WAIT_CYCLES_WIDTH-1:0] wake_up_delay,
    input [WAIT_CYCLES_WIDTH-1:0] power_up_delay,

    output reg memories_ready,
    output reg in_power_down
);
    // TODO: could investigate whether counting up is easy to implement and saves resources

    reg [WAIT_CYCLES_WIDTH-1:0] wake_up_delay_counter;
    reg [WAIT_CYCLES_WIDTH-1:0] power_up_delay_counter;

    always @(posedge clk) begin
        if (rst) begin
            wake_up_delay_counter <= 0;
            power_up_delay_counter <= 0;
            in_power_down <= 1'b0;
            memories_ready <= 1'b1;
        end else if (wake_up_delay_counter != 0) begin
            if (wake_up_delay_counter == 1)
                memories_ready <= 1'b1;

            wake_up_delay_counter <= wake_up_delay_counter - 1;
        end else if (power_up_delay_counter != 0) begin
            power_up_delay_counter <= power_up_delay_counter - 1;
        end else if (in_power_down == 1'b1 && should_be_on && power_up_delay_counter == 0) begin
            in_power_down <= 1'b0;
            wake_up_delay_counter <= wake_up_delay;
        end else if (allow_power_down && !should_be_on && memories_ready == 1'b1) begin
            in_power_down <= 1'b1;
            memories_ready <= 1'b0;
            power_up_delay_counter <= power_up_delay;
        end
    end
endmodule
