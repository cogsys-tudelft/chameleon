module input_memory #(
    parameter int BLOCKS_KERNEL_WIDTH = 8,
    parameter int BLOCKS_WIDTH = 4,
    parameter int NUM_ROWS = 32,
    parameter int WIDTH = 32,
    parameter int SUBCHUNK_WIDTH = 8,
    parameter int START_ADDRESS_BIT_WIDTH = 14,
    parameter int MESSAGE_BIT_WIDTH = 32,
    parameter int ACTIVATION_WORD_BIT_WIDTH = 64,
    parameter int ACTIVATION_BIT_WIDTH = 4,
    localparam int AddressBitWidth = $clog2(NUM_ROWS)
) (
    input clk,
    input rst,

    input enable,
    input restart,

    input [AddressBitWidth-1:0] address,
    input read_enable,

    input write_new,
    input read_sync,
    input code_is_input,

    input [AddressBitWidth-1:0] spi_address,
    input [MESSAGE_BIT_WIDTH-1:0] input_spi_data_in,
    output [MESSAGE_BIT_WIDTH-1:0] input_spi_data_out,

    // Configuration wires
    input fill_first,
    input require_single_chunk,
    input use_subchunks,
    input load_inputs_from_activation_memory,

    // Control wires
    input in_idle,
    input running,

    input in_first_block,
    input in_last_block_of_input_layer,
    input is_layer_residual,
    input [BLOCKS_WIDTH-1:0] input_blocks,
    input [BLOCKS_KERNEL_WIDTH-1:0] input_blocks_times_kernel_size,

    input  data_available,
    output data_required,

    output reg ready,

    input [WIDTH-1:0] data_in,
    output [ACTIVATION_WORD_BIT_WIDTH-1:0] data_out
);
    // Local parameters ---------------------------------------------------------------------------

    localparam int ChunksCompletedCounterWidth = $clog2(ACTIVATION_WORD_BIT_WIDTH / WIDTH);
    localparam int SubchunksCompletedCounterWidth = $clog2(WIDTH / SUBCHUNK_WIDTH);
    localparam int SubchunksCompletedCounterWidthMin1 = SubchunksCompletedCounterWidth == 0 ? 1 : SubchunksCompletedCounterWidth;

    // Check parameters ---------------------------------------------------------------------------

    if ((ACTIVATION_WORD_BIT_WIDTH + WIDTH - 1) / WIDTH != ACTIVATION_WORD_BIT_WIDTH / WIDTH) begin : gen_raise_error__WIDTH_must_be_perfectly_divisible_by_MESSAGE_BIT_WIDTH
        $fatal(1, "ERROR: WIDTH must be perfectly divisible by ACTIVATION_WORD_BIT_WIDTH");
    end

    if ((WIDTH + SUBCHUNK_WIDTH - 1) / SUBCHUNK_WIDTH != WIDTH / SUBCHUNK_WIDTH) begin : gen_raise_error__WIDTH_must_be_perfectly_divisible_by_SUBCHUNK_WIDTH
        $fatal(1, "ERROR: WIDTH must be perfectly divisible by SUBCHUNK_WIDTH");
    end

    if (WIDTH > ACTIVATION_WORD_BIT_WIDTH) begin : gen_raise_error__WIDTH_must_be_smaller_than_ACTIVATION_WORD_BIT_WIDTH
        $fatal(1, "ERROR: WIDTH must be smaller than ACTIVATION_WORD_BIT_WIDTH");
    end

    if (WIDTH < ACTIVATION_BIT_WIDTH) begin: gen_raise_error__WIDTH_must_be_larger_than_ACTIVATION_BIT_WIDTH
        $fatal(1, "ERROR: WIDTH must be larger than ACTIVATION_BIT_WIDTH");
    end

    // Registers ----------------------------------------------------------------------------------

    reg [AddressBitWidth-1:0] oldest_data_address;
    reg [AddressBitWidth-1:0] last_written_address;
    reg [ChunksCompletedCounterWidth-1:0] chunks_completed;
    reg [SubchunksCompletedCounterWidthMin1-1:0] subchunks_completed;

    reg [AddressBitWidth-1:0] num_blocks_required;

    reg load_configuration_after_exiting_idle;

    // Wires --------------------------------------------------------------------------------------

    wire blocks_required = num_blocks_required != 0;
    wire request_new_blocks = read_enable && in_last_block_of_input_layer && address == last_written_address;
    wire chunk_completed_with_subchunks = (use_subchunks ? (subchunks_completed == 2 ** SubchunksCompletedCounterWidth-1) : 1'b1);
    wire detect_not_ready = read_enable && in_first_block && address == last_written_address && !is_layer_residual;
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] input_memory_data_in = data_in << (WIDTH * chunks_completed + SUBCHUNK_WIDTH * subchunks_completed);
    // Write with mask 1111111... by default to make sure that all the leading bits are set to 0 and dont stay unknown
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] input_memory_mask = {{ACTIVATION_WORD_BIT_WIDTH{1'b1}}} << (WIDTH * chunks_completed + SUBCHUNK_WIDTH * subchunks_completed);

    // Combinational output logic -----------------------------------------------------------------

    assign data_required = blocks_required && enable;

    // Child modules ------------------------------------------------------------------------------

    managed_input_memory #(
        .INPUT_WORD_BIT_WIDTH(ACTIVATION_WORD_BIT_WIDTH),
        .INPUT_ROWS(NUM_ROWS),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) managed_input_memory_inst (
        .clk(clk),

        .write_new(write_new),
        .read_sync(read_sync),
        .code_is_input(code_is_input),

        .spi_address(spi_address),
        .input_spi_data_in(input_spi_data_in),
        .input_spi_data_out(input_spi_data_out),

        .input_control_read_enable(enable & read_enable),
        .input_control_write_enable(enable & (~(restart | load_configuration_after_exiting_idle)) & data_available),
        .input_control_address_read(address),
        .input_control_address_write(oldest_data_address),
        .input_control_data_in(input_memory_data_in),
        .input_control_mask(input_memory_mask),

        .input_data_out(data_out)
    );

    // Sequential logic ---------------------------------------------------------------------------

    // TODO: make this cleaner!
    // load_configuration_after_exiting_idle will be 1 in the first cycle of being in state 1, after that it will
    // be 0. Since the enable is only high at minimum when the state is 1, we use this signal as a
    // one cycle toggle to load the input_blocks_times_kernel_size from the configuration memory.
    // This implementation can definitely be cleaner, but it works for now.
    always @(posedge clk) begin
        if (rst | in_idle) begin
            load_configuration_after_exiting_idle <= 1'b1;
        end else if (running && load_configuration_after_exiting_idle == 1'b1) begin
            load_configuration_after_exiting_idle <= 1'b0;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            num_blocks_required <= 0;
            ready <= 1'b0;
            chunks_completed <= 0;
            subchunks_completed <= 0;
        end else if ((enable & (restart | load_configuration_after_exiting_idle)) | in_idle) begin
            // If we are going to process a new example (set of activations representing one class)
            // The | in_idle is needed as when continuous processing took place before the current
            // forward pass, there will not have been a restart signal fired when the continuous processing
            // got killed.

            if (load_inputs_from_activation_memory & load_configuration_after_exiting_idle) begin
                ready <= 1'b1;
            end else begin
                // Setting num_blocks_required to a non-zero value, wil trigger the high speed input port to be enabled
                // Cannot do this in the reset, as we first need to wait till input_blocks_times_kernel_size is set
                num_blocks_required <= input_blocks_times_kernel_size;
                ready <= 1'b0;
            end

            oldest_data_address <= 0;

            // Data available can never be true at the same time as restart, as
        end else if (enable & ~load_inputs_from_activation_memory) begin
            if (data_available) begin
                if ((chunks_completed == 2**ChunksCompletedCounterWidth-1 & chunk_completed_with_subchunks) | require_single_chunk) begin
                    chunks_completed <= 0;
                    subchunks_completed <= 0;

                    if (fill_first) begin
                        // Only continue when this was the last block we needed to fill the memory
                        if (num_blocks_required == 1) ready <= 1'b1;
                    end else begin
                        // Always continue processing when we have new data available
                        // and if we dont want to fill the memory first
                        ready <= 1'b1;
                    end

                    num_blocks_required <= num_blocks_required - 1;
                    last_written_address <= oldest_data_address;

                    // Line below basically implements a circular buffer for the oldest data address
                    oldest_data_address <= oldest_data_address == input_blocks_times_kernel_size - 1 ? 0 : oldest_data_address + 1;
                end else begin
                    if (use_subchunks) subchunks_completed <= subchunks_completed + 1;

                    if (chunk_completed_with_subchunks) chunks_completed <= chunks_completed + 1;

                    if (detect_not_ready) ready <= 1'b0;
                end
            end else if (blocks_required) begin
                // Effectively: if !restart && !data_available but we still need data
                // If the PE array control tries to read the data that was just written, we can safely assume that
                // the next data is not written yet, so we have to stop processing. However, in case the reading
                // is happening is a residual layer, we dont have to unready the input memory just yet, as the
                // next clockcycle will not need data from the input memory.
                if (detect_not_ready) ready <= 1'b0;
            end else if (request_new_blocks) begin
                // If we are in the last block of the input and we are reading the oldest data, we can assume
                // that we need to start overwriting the oldest data.
                // If !restart && !data_available && !blocks_required and we currently don't need data
                num_blocks_required <= input_blocks + 1;
            end
        end
    end

endmodule
