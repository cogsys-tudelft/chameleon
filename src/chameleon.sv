`include "states.vh"

module chameleon #(
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
    parameter int WEIGHT_ROWS = 1024, // Needs to be twice as large as actual rows to retrieve subsection weights correctly
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
    input  is_new_task, // could remove is_new_task by setting it via SPI
    output in_idle,

    input  SCK, // can be combined with in_request
    output MISO, // can be combined with clk_int_div
    input  MOSI, // can be combined with in_acknowledge

    output clk_int_div,

    input [HIGH_SPEED_IN_PINS-1:0] data_in,
    input in_request,
    output out_acknowledge,

    output [HIGH_SPEED_OUT_PINS-1:0] data_out,
    output out_request,
    input in_acknowledge
);

    // ============================================================================================
    // Parameter checks
    // ============================================================================================

    if (MAX_NUM_LOGITS > MAX_NUM_CHANNELS) begin : gen_raise_error__max_num_logits_cannot_be_larger_than_max_num_channels
        $fatal(1, "ERROR: MAX_NUM_LOGITS cannot be larger than MAX_NUM_CHANNELS");
    end

    if ((PE_COLS * ACTIVATION_BIT_WIDTH / HIGH_SPEED_OUT_PINS) * HIGH_SPEED_OUT_PINS != PE_COLS * ACTIVATION_BIT_WIDTH) begin : gen_raise_error__total_activation_data_should_be_multiple_of_high_speed_out_pins
        $fatal(1, "ERROR: Total activation data should be multiple of high speed out pins");
    end

    // ============================================================================================
    // Computed parameters
    // ============================================================================================

    localparam int ArgmaxIndexWidth = $clog2(MAX_NUM_LOGITS);
    localparam int SHOT_BIT_WIDTH = $clog2(MAX_SHOTS+1);
    localparam int FEW_SHOT_SHIFT_BIT_WIDTH = $clog2(PE_COLS);

    localparam int ACTIVATION_WORD_BIT_WIDTH = PE_ROWS * ACTIVATION_BIT_WIDTH;
    localparam int SUBSECTION_ACTIVATION_WORD_BIT_WIDTH = SUBSECTION_SIZE * ACTIVATION_BIT_WIDTH;
    localparam int WeightWordBitWidth = PE_ROWS * PE_COLS * WEIGHT_BIT_WIDTH;
    localparam int BiasWordBitWidth = PE_COLS * BIAS_BIT_WIDTH;

    localparam int ACTIVATION_ADDRESS_WIDTH = $clog2(ACTIVATION_ROWS);
    localparam int WEIGHT_ADDRESS_WIDTH = $clog2(WEIGHT_ROWS);
    localparam int BIAS_ADDRESS_WIDTH = $clog2(BIAS_ROWS);
    localparam int InputAddressWidth = $clog2(INPUT_ROWS);

    localparam int MaxNumBlocks = MAX_NUM_CHANNELS / PE_ROWS;
    localparam int BLOCKS_WIDTH = $clog2(MaxNumBlocks);
    localparam int KERNEL_WIDTH = $clog2(MAX_KERNEL_SIZE+1);
    localparam int BLOCKS_KERNEL_WIDTH = $clog2(MaxNumBlocks * MAX_KERNEL_SIZE);
    localparam int LAYER_WIDTH = $clog2(MAX_NUM_LAYERS);
    localparam int CumsumWidth = ACTIVATION_ADDRESS_WIDTH;

    localparam int MaxWeightValue = 2**(2**(WEIGHT_BIT_WIDTH-1)-1);
    localparam int MaxActivationValue = 2**ACTIVATION_BIT_WIDTH-1;
    localparam int FLog2Width = $clog2(MaxWeightValue)+1;
    localparam int LeftShiftFewShotScaleWidth = $clog2(FLog2Width);
    localparam int RightShiftFewShotScaleWidth = $clog2(MaxActivationValue*MAX_SHOTS/MaxWeightValue);
    localparam int FewShotScaleWidth = 1 + ((LeftShiftFewShotScaleWidth > RightShiftFewShotScaleWidth) ? LeftShiftFewShotScaleWidth : RightShiftFewShotScaleWidth);
    localparam int KShotDivisionScaleWidth = $clog2($clog2(MAX_SHOTS));

    localparam int StateBitWidth = $clog2(`NUMBER_OF_STATES);

    // At maximum, we want to be able to send all blocks, in pieces of HIGH_SPEED_OUT_PINS bits
    localparam int SentCounterBitWidthForBlocks = $clog2(ACTIVATION_WORD_BIT_WIDTH / HIGH_SPEED_OUT_PINS + 1);
    localparam int SentCounterBitWidthForArgmax = $clog2((ArgmaxIndexWidth + ACCUMULATION_BIT_WIDTH) / HIGH_SPEED_OUT_PINS + 1);
    localparam int SentCounterBitWidth = (SentCounterBitWidthForBlocks > SentCounterBitWidthForArgmax) ? SentCounterBitWidthForBlocks : SentCounterBitWidthForArgmax;

    localparam int OnesLengthWeight = ACTIVATION_WORD_BIT_WIDTH - SUBSECTION_ACTIVATION_WORD_BIT_WIDTH;
    localparam int SubsectionWeightWordBitWidth = WEIGHT_BIT_WIDTH * SUBSECTION_SIZE * PE_COLS;
    localparam int RestWeightWordBitWidth = WEIGHT_BIT_WIDTH * (PE_ROWS - SUBSECTION_SIZE) * PE_COLS;

    localparam int NumberOfMostNegativeBiases = PE_COLS - 1;
    localparam int NumberOfZeroesInMostNegativeBias = BIAS_BIT_WIDTH - 1;

    localparam int LeftShiftWidth = $clog2(ARGMAX_INPUTS_SHIFT+1) == 0 ? 1 : $clog2(ARGMAX_INPUTS_SHIFT+1);

    // ============================================================================================
    // Wires and registers
    // ============================================================================================

    // Clock and reset wires ----------------------------------------------------------------------

    wire clk, clk_int;
    wire rst_sync;

    wire enable_clock_divider;

    wire clk_div_in;
    wire clk_div_out;

    // FSM wires ----------------------------------------------------------------------------------

    wire continuous_processing;
    wire toggle_processing_new;

    wire start_few_shot_processing;
    wire done_few_shot_processing;

    wire skip_sending;

    wire [StateBitWidth-1:0] state;
    wire [StateBitWidth-1:0] next_state;

    wire start_sending;

    wire processing_few_shot;
    wire running;
    wire sending;

    // SPI client wires ---------------------------------------------------------------------------

    wire [CODE_BIT_WIDTH-1:0] code;

    wire [START_ADDRESS_BIT_WIDTH-1:0] spi_address;

    reg [MESSAGE_BIT_WIDTH-1:0] MISO_data;
    wire [MESSAGE_BIT_WIDTH-1:0] MOSI_data;

    // Wires per code -----------------------------------------------------------------------------

    // 0: config (not readable)
    // 0: pointers (not writable)
    wire code_is_pointers = (code == 0);
    wire code_is_weight = (code == 1);
    wire code_is_bias = (code == 2);
    wire code_is_activation = (code == 3);
    wire code_is_input = (code == 4);

    // SPI clock barrier crossing wires ------------------------------------------------------------

    // TODO: give these two signals better names
    wire write_new;
    wire read_sync;

    // Config memory wires ------------------------------------------------------------------------

    wire config_data_ready;
    wire [START_ADDRESS_BIT_WIDTH-1:0] current_config_address;
    wire [MESSAGE_BIT_WIDTH-1:0] config_data;

    wire classification;
    wire continued_learning;
    wire power_down_memories_while_running;
    wire fill_input_memory;
    wire force_downsample;
    wire power_down_srams_in_standby;
    wire in_4x4_mode;
    wire power_down_small_bias;
    wire require_single_chunk;
    wire use_subchunks;
    wire in_context_learning;
    wire send_all_argmax_chunks;
    wire load_inputs_from_activation_memory;

    wire [WEIGHT_ADDRESS_WIDTH-1:0] max_weight_address;
    wire [BIAS_ADDRESS_WIDTH-1:0] max_bias_address;
    wire [SHOT_BIT_WIDTH-1:0] shots;
    wire shift_few_shot_data_right;
    wire [LeftShiftFewShotScaleWidth-1:0] left_shift_few_shot_scale;
    wire [RightShiftFewShotScaleWidth-1:0] right_shift_few_shot_scale;
    wire [FewShotScaleWidth-1:0] few_shot_scale;
    wire [KShotDivisionScaleWidth-1:0] k_shot_division_scale;
    wire use_l2_for_few_shot;

    wire [LAYER_WIDTH-1:0] num_conv_layers;
    wire [LAYER_WIDTH-1:0] num_conv_and_linear_layers;
    wire [LAYER_WIDTH-1:0] num_conv_and_linear_layers_full_icl_net;

`ifdef SYNTHESIZE_FOR_SILICON
    wire [6-1:0] ring_oscillator_stage_selection;
`endif

    wire [KERNEL_WIDTH-1:0] kernel_size_per_layer[MAX_NUM_LAYERS:0];
    wire [BLOCKS_WIDTH-1:0] blocks_per_layer[MAX_NUM_LAYERS:0];
    wire [CumsumWidth-1:0] blocks_per_layer_times_kernel_size_cumsum[MAX_NUM_LAYERS:0];
    wire [BLOCKS_KERNEL_WIDTH-1:0] input_blocks_times_kernel_size;
    wire [BLOCKS_KERNEL_WIDTH-1:0] input_blocks_times_kernel_size_icl_head;
    wire [2*SCALE_BIT_WIDTH-1:0] scale_and_residual_scale_per_layer[MAX_NUM_LAYERS-1:0];

    // Pointers wires -----------------------------------------------------------------------------

    wire [MESSAGE_BIT_WIDTH-1:0] pointers_spi_data_out;

    // High-speed in bus wires --------------------------------------------------------------------

    wire data_required;
    wire data_available;

    reg pass_data_available_and_required;
    wire intercepted_data_required;
    wire intercepted_data_available;

    // Input memory wires -------------------------------------------------------------------------

    wire enable_input_memory;
    wire restart_input_memory;

    wire is_input_memory_ready;

    reg handling_sending_inverted;

    wire disable_input_memory_to_send_data_out;

    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] input_data_in;

    wire in_first_block;
    wire in_last_block_of_input_layer;

    wire [MESSAGE_BIT_WIDTH-1:0] input_spi_data_out;

    // PE array control wires ---------------------------------------------------------------------

    wire enable_pe_control;
    wire restart_pe_array_control;

    wire enable_pe_array;

    wire apply_identity;
    wire dont_load_weights;

    wire dont_wait_for_linear_layer;
    wire apply_in_scale;

    wire is_input_layer_pe_control;

    wire is_input_layer;
    wire feed_input_data_to_pe_array;
    wire is_layer_residual;
    wire is_output_layer;
    wire in_cycle_before_sending_embedding_out;
    wire in_last_cycle_of_address_generation;
    wire processed_all_input_blocks_into_current_block_out;
    wire [LAYER_WIDTH-1:0] layer;
    wire [LAYER_WIDTH-1:0] delayed_layer;

    wire output_ready;

    wire load_scale_and_bias;

    wire [ACTIVATION_ADDRESS_WIDTH-1:0] in_activation_address;
    wire [ACTIVATION_ADDRESS_WIDTH-1:0] out_activation_address;

    wire [WEIGHT_ADDRESS_WIDTH-1:0] weight_address_pe_control;
    wire [BIAS_ADDRESS_WIDTH-1:0] bias_address_pe_control;

    wire output_ready_in_output_layer;
    wire output_ready_for_argmax;

    // PE array wires -----------------------------------------------------------------------------

    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] in_activation_data;
    wire [ACTIVATION_BIT_WIDTH-1:0] in_activation_data_vector[PE_ROWS];
    wire [WEIGHT_BIT_WIDTH-1:0] weight_data_out_vector[PE_ROWS][PE_COLS];
    wire [ACTIVATION_BIT_WIDTH-1:0] out_activation_data_vector[PE_COLS];

    wire enable_pe_array_few_shot_corrected;

    wire signed [ACCUMULATION_BIT_WIDTH-1:0] col_accumulator[PE_COLS];

    wire scale_and_bias_ready;
    wire [WEIGHT_BIT_WIDTH-1:0] weights[PE_ROWS][PE_COLS];
    wire [BIAS_BIT_WIDTH-1:0] biases[PE_COLS];
    wire [SCALE_BIT_WIDTH-1:0] in_scale, out_scale;
    wire [SCALE_BIT_WIDTH-1:0] in_scale_few_shot_corrected;

    // Few-shot control wires ---------------------------------------------------------------------

    wire is_new_task_sync;
    wire processing_new_task_samples;
    wire enable_pe_array_during_few_shot;
    wire scale_and_bias_ready_few_shot;
    wire use_scale_and_bias;
    wire enable_activation_memory_for_few_shot;
    wire enable_weight_memory_for_few_shot;
    wire configured_for_few_shot_processing;
    wire use_few_shot_next_layer_blocks;

    wire enable_bias_memory_for_few_shot;

    wire [WEIGHT_ADDRESS_WIDTH-1:0] weight_address_few_shot_corrected;
    wire [BIAS_ADDRESS_WIDTH-1:0] bias_address_few_shot_corrected;

    wire [FEW_SHOT_SHIFT_BIT_WIDTH-1:0] few_shot_shift_required;

    wire [LAYER_WIDTH-1:0] num_conv_and_linear_layers_few_shot_corrected;

    wire [BLOCKS_WIDTH-1:0] blocks_per_layer_few_shot_corrected;

    wire [ArgmaxIndexWidth-FEW_SHOT_SHIFT_BIT_WIDTH-1:0] weight_bias_offset;
    wire [ArgmaxIndexWidth-1:0] ways_received;
    wire [SHOT_BIT_WIDTH-1:0] shots_received;

    wire [LeftShiftFewShotScaleWidth-1:0] new_few_shot_weights [PE_COLS];
    wire [WEIGHT_BIT_WIDTH*PE_COLS-1:0] new_few_shot_weights_flattened;

    wire [FLog2Width-1:0] col_accumulator_lsbs [PE_COLS];

    wire [LAYER_WIDTH-1:0] embedder_layer_index;

    // Squared-sum accumulator wires --------------------------------------------------------------

    wire [BIAS_BIT_WIDTH-1:0] new_few_shot_bias;
    wire restart_squared_sum_accumulator;

    // Argmax wires -------------------------------------------------------------------------------

    wire enable_argmax;
    wire restart_argmax;

    wire [ArgmaxIndexWidth-1:0] argmax;

    wire [ACCUMULATION_BIT_WIDTH*PE_COLS-1:0] col_accumulator_vector_argmax;
    wire signed [ACCUMULATION_BIT_WIDTH-1:0] data_gated_col_accumulator_vector_argmax [PE_COLS];

    wire signed [ACCUMULATION_BIT_WIDTH-1:0] max_out_value;

    // High-speed out bus wires -------------------------------------------------------------------

    wire [HIGH_SPEED_OUT_PINS-1:0] data_to_output;
    wire [SentCounterBitWidth-1:0] num_sends;

    wire [SentCounterBitWidth-1:0] sent_counter;

    wire done_sending;

    // Global SRAM power-down controller wires ----------------------------------------------------

    wire memories_ready;
    wire global_power_down;
    reg keep_memories_powered_up;
    wire [WAIT_CYCLES_WIDTH-1:0] wake_up_delay;
    wire [WAIT_CYCLES_WIDTH-1:0] power_up_delay;

    wire power_down_memories_while_running_sync;
    wire power_down_srams_in_standby_sync;

    // Weight memory wires ------------------------------------------------------------------------

    wire [WEIGHT_ADDRESS_WIDTH-1:0] weight_control_address;
    wire [WeightWordBitWidth-1:0] weight_control_data_in;
    wire [WeightWordBitWidth-1:0] weight_data_out;
    wire [WeightWordBitWidth-1:0] weight_control_mask;
    wire weight_control_chip_select;
    wire weight_control_write_enable;

    wire [MESSAGE_BIT_WIDTH-1:0] weight_spi_data_out;

    wire [SubsectionWeightWordBitWidth-1:0] subsection_weight_mask;
    wire [RestWeightWordBitWidth-1:0] rest_weight_mask;

    wire [SubsectionWeightWordBitWidth-1:0] subsection_weight_data;
    wire [RestWeightWordBitWidth-1:0] rest_weight_data;


    // Bias memory wires --------------------------------------------------------------------------

    wire [BIAS_ADDRESS_WIDTH-1:0] bias_control_address;
    wire [BiasWordBitWidth-1:0] bias_control_data_in;
    wire [BiasWordBitWidth-1:0] bias_data_out;
    wire [BiasWordBitWidth-1:0] bias_control_mask;
    wire bias_control_chip_select;
    wire bias_control_write_enable;

    wire [MESSAGE_BIT_WIDTH-1:0] bias_spi_data_out;

    // Activation memory wires --------------------------------------------------------------------

    wire [ACTIVATION_ADDRESS_WIDTH-1:0] in_activation_control_address;
    wire [ACTIVATION_ADDRESS_WIDTH-1:0] out_activation_control_address;
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_control_data_in;
    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_control_mask;
    wire activation_control_read_enable;
    wire activation_control_write_enable;

    wire [ACTIVATION_WORD_BIT_WIDTH-1:0] activation_data_out;
    wire [MESSAGE_BIT_WIDTH-1:0] activation_spi_data_out;

    // ============================================================================================
    // Modules
    // ============================================================================================

    // Clock --------------------------------------------------------------------------------------

`ifdef SYNTHESIZE_FOR_SILICON
    clock_generator_chameleon clock_generator_chameleon_inst (
        .enable(enable_clk_int),
        .stage_selection(ring_oscillator_stage_selection),
        .clk_out(clk_int)
    );
`else
    assign clk_int = 1'b0;
`endif

    // and the input into the clock divider to save power when we don't need it
    assign clk_div_in = clk_int & enable_clock_divider;

    clock_divider #(
        .NUM_STAGES(CLOCK_DIVIDER_STAGES)
    ) clock_divider_inst (
        .clk(clk_div_in),
        .rst(rst_async),
        .clk_div(clk_div_out)
    );

    // and the output of the clock divider so we can safely connect this and more signals to a single pad
    // and-ing the input signal is not enough, since there are NUM_STAGES registers in between the input
    // and the output of the clock divider
    assign clk_int_div = clk_div_out & enable_clock_divider;

    // In case internal clock has issues, we can always provide an external clock
    ext_or_int_clock ext_or_int_clock_inst (
        .clk_ext(clk_ext),
        .clk_int(clk_int),
        .clk(clk)
    );

    // Make external asynchronous reset synchronous with the internal clock
    double_flop_synchronizer #(
        .AT_POSEDGE_RST(0)
    ) double_flop_synchronizer_rst (
        .clk(clk),
        .rst(1'b0),

        .enable(1'b1),

        .in (rst_async),
        .out(rst_sync)
    );

    // FSM ----------------------------------------------------------------------------------------

    assign in_idle = state == `IDLE;
    assign processing_few_shot = state == `PROCESSING_FEW_SHOT;
    assign sending = state == `SENDING;
    assign running = state == `RUNNING;

    triple_flop_toggle_synchronizer triple_flop_toggle_synchronizer_processing_new (
        .clk(clk),
        .rst(rst_sync),

        .enable(1'b1),

        .in (toggle_processing),
        .out(toggle_processing_new)
    );

    fsm fsm_inst (
        .clk(clk),
        .rst(rst_sync),

        .continuous_processing(continuous_processing),
        .classification(classification),
        // When there is only one linear layer, we check that weight_address_pe_control != 0
        // to make sure that we still return to running as long as we have not reset the weight
        // address to zero, indicating that all outputs have been computed
        .is_output_layer(num_conv_and_linear_layers == 1 ? weight_address_pe_control != 0 : is_output_layer),

        .toggle_processing_new(toggle_processing_new),
        .start_sending(start_sending),
        .done_sending(done_sending),

        .skip_sending(skip_sending),
        .start_few_shot_processing(start_few_shot_processing),
        .done_few_shot_processing(done_few_shot_processing),

        .state(state),
        .next_state(next_state)
    );

    // SPI client --------------------------------------------------------------------------------

    spi_client #(
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH),
        .CODE_BIT_WIDTH(CODE_BIT_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH)
    ) spi_client_inst (
        .rst_async(rst_async),

        .SCK(SCK),
        .MISO(MISO),
        .MOSI(MOSI),

        .clk(clk),
        .rst(rst_sync),

        .enable_configuration(in_idle),

        .code(code),
        .current_address(spi_address),

        .MISO_data(MISO_data),
        .MOSI_data(MOSI_data),

        .config_data_ready(config_data_ready),
        .current_config_address(current_config_address),
        .config_data(config_data),

        .write_new(write_new),
        .read_sync(read_sync)
    );

    always @(posedge clk) begin
        // We are now writing for multiple clockcyles to MISO_data as read_sync stays high for multiple clock cycles
        // in the clk clock domain (while only 1 in the SCK domain). This is to make sure that the data in MISO_data
        // stays correct until the negative edge of the SPI. For example, if we only read for one clock cycle, maybe
        // the data in MISO_data is 0 again when the next negative edge of the SPI comes.
        if (read_sync) begin
            if (code_is_pointers) begin
                MISO_data <= pointers_spi_data_out;
            end else if (code_is_weight) begin
                MISO_data <= weight_spi_data_out;
            end else if (code_is_bias) begin
                MISO_data <= bias_spi_data_out;
            end else if (code_is_activation) begin
                MISO_data <= activation_spi_data_out;
            end else if (code_is_input) begin
                MISO_data <= input_spi_data_out;
            end
        end
    end

    // Config memory ------------------------------------------------------------------------------

    // Disable lint for ring_oscillator_stage_selection parameter
    config_memory #(
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .BLOCKS_WIDTH(BLOCKS_WIDTH),
        .BLOCKS_KERNEL_WIDTH(BLOCKS_KERNEL_WIDTH),
        .CUMSUM_WIDTH(CumsumWidth),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .LAYER_WIDTH(LAYER_WIDTH),
        .SHOT_BIT_WIDTH(SHOT_BIT_WIDTH),
        .FEW_SHOT_SCALE_WIDTH(FewShotScaleWidth),
        .K_SHOT_SCALE_WIDTH(KShotDivisionScaleWidth),
        .BIAS_ADDRESS_WIDTH(BIAS_ADDRESS_WIDTH),
        .WEIGHT_ADDRESS_WIDTH(WEIGHT_ADDRESS_WIDTH),
        .SCALE_BIT_WIDTH(SCALE_BIT_WIDTH),
        .WAIT_CYCLES_WIDTH(WAIT_CYCLES_WIDTH)
    ) cfg (
        .SCK(SCK),
        .rst_async(rst_async),

        .config_data_ready(config_data_ready),
        .current_config_address(current_config_address),  // TODO: can reduce the address bitwidth accordingly here
        .config_spi_data_in(config_data),

        // We do this extra assign instead of using cfg.enable_clock_divider directly in the clock section
        // of the code, as the config memory is only declared later and causes issues in Genus synthesis.
        .enable_clock_divider (enable_clock_divider),
        .continuous_processing(continuous_processing),

`ifdef SYNTHESIZE_FOR_SILICON
        .ring_oscillator_stage_selection(ring_oscillator_stage_selection),
`else
        .ring_oscillator_stage_selection(),
`endif

        .classification(classification),
        .continued_learning(continued_learning),
        .power_down_memories_while_running(power_down_memories_while_running),
        .fill_input_memory(fill_input_memory),
        .force_downsample(force_downsample),
        .power_down_small_bias(power_down_small_bias),
        .power_down_srams_in_standby(power_down_srams_in_standby),
        .send_all_argmax_chunks(send_all_argmax_chunks),
        .in_4x4_mode(in_4x4_mode),
        .require_single_chunk(require_single_chunk),
        .use_subchunks(use_subchunks),
        .wake_up_delay(wake_up_delay),
        .power_up_delay(power_up_delay),
        .in_context_learning(in_context_learning),
        .num_conv_and_linear_layers_full_icl_net(num_conv_and_linear_layers_full_icl_net),
        .load_inputs_from_activation_memory(load_inputs_from_activation_memory),

        .max_weight_address(max_weight_address),
        .max_bias_address(max_bias_address),
        .shots(shots),
        .few_shot_scale(few_shot_scale),
        .k_shot_division_scale(k_shot_division_scale),
        .use_l2_for_few_shot(use_l2_for_few_shot),
        .input_blocks_times_kernel_size_icl_head(input_blocks_times_kernel_size_icl_head),

        .input_blocks_times_kernel_size(input_blocks_times_kernel_size),
        .num_conv_layers(num_conv_layers),
        .num_conv_and_linear_layers(num_conv_and_linear_layers),
        .kernel_size_per_layer(kernel_size_per_layer),
        .blocks_per_layer(blocks_per_layer),
        .blocks_per_layer_times_kernel_size_cumsum(blocks_per_layer_times_kernel_size_cumsum),
        .scale_and_residual_scale_per_layer(scale_and_residual_scale_per_layer)
    );

    // Pointers -----------------------------------------------------------------------------------

    pointers #(
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),

        .WAYS_BIT_WIDTH(ArgmaxIndexWidth),
        .SHOT_BIT_WIDTH(SHOT_BIT_WIDTH),
        .FEW_SHOT_SHIFT_BIT_WIDTH(FEW_SHOT_SHIFT_BIT_WIDTH)
    ) pointers_inst (
        .clk(clk),

        .read_sync(read_sync),
        .code_is_pointers(code_is_pointers),

        .ways_received(ways_received),
        .shots_received(shots_received),
        .configured_for_few_shot_processing(configured_for_few_shot_processing),
        .weight_bias_offset(weight_bias_offset),
        .few_shot_shift_required(few_shot_shift_required),

        .spi_address(spi_address),
        .pointer_spi_data_out(pointers_spi_data_out)
    );

    // High-speed in bus --------------------------------------------------------------------------

    high_speed_in_bus high_speed_in_bus_inst (
        .clk(clk),
        .rst(rst_sync),

        .data_required (intercepted_data_required),
        .data_available(data_available),

        .request(in_request),
        .acknowledge(out_acknowledge)
    );

    // Enable and restart signals -----------------------------------------------------------------

    // Handling sending is low after we generate the last address and high after we send the last data
    // However, since we only need the inverted value of this, only store and compute that.
    always @(posedge clk) begin
        if (rst_sync | in_idle) begin
            handling_sending_inverted <= 1'b1;
        end else if (handling_sending_inverted == 1'b0 & done_sending) begin
            handling_sending_inverted <= 1'b1;
        end else if (disable_input_memory_to_send_data_out) begin
            handling_sending_inverted <= 1'b0;
        end
    end

    assign disable_input_memory_to_send_data_out = ~processing_new_task_samples & ((in_last_cycle_of_address_generation & classification) | (~classification & in_cycle_before_sending_embedding_out));

    // TODO: is this correct? put it down earlier
    assign enable_input_memory = running & handling_sending_inverted;
    assign restart_input_memory = in_last_cycle_of_address_generation & ~continuous_processing;

    assign enable_pe_control = memories_ready & is_input_memory_ready & dont_wait_for_linear_layer & enable_input_memory;
    assign restart_pe_array_control = in_idle;

    assign output_ready_in_output_layer = output_ready & is_output_layer;
    assign output_ready_for_argmax = output_ready_in_output_layer & classification;
    assign enable_argmax = output_ready_for_argmax & ~processing_new_task_samples & running;
    assign restart_argmax = done_sending;

    assign skip_sending = ((output_ready_for_argmax & processing_new_task_samples) | (output_ready_in_output_layer & processing_new_task_samples)) & ~enable_pe_array;

    // Below couple of lines are logic from the delay start controller, which is gone now but these lines are
    // still necessary to make the chip run more than one forward pass
    assign intercepted_data_available = pass_data_available_and_required ? data_required : data_available;
    assign intercepted_data_required = pass_data_available_and_required ? 1'b0 : data_required;

    always @(posedge clk) begin
        if (rst_sync | in_idle) begin
            pass_data_available_and_required <= 1'b1;
        end else if (running & pass_data_available_and_required) begin
            pass_data_available_and_required <= 1'b0;
        end
    end

    // Input memory -------------------------------------------------------------------------------

    input_memory #(
        .BLOCKS_KERNEL_WIDTH(BLOCKS_KERNEL_WIDTH),
        .BLOCKS_WIDTH(BLOCKS_WIDTH),
        .NUM_ROWS(INPUT_ROWS),
        .WIDTH(HIGH_SPEED_IN_PINS),
        .SUBCHUNK_WIDTH(HIGH_SPEED_OUT_PINS),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH),
        .ACTIVATION_WORD_BIT_WIDTH(ACTIVATION_WORD_BIT_WIDTH),
        .ACTIVATION_BIT_WIDTH(ACTIVATION_BIT_WIDTH)
    ) input_memory_inst (
        .clk(clk),
        .rst(rst_sync),

        .enable (enable_input_memory),
        .restart(restart_input_memory),

        // We and with enable_pe_control as is_input_layer is enabled long before the PE array actually reads data from the input memory
        // The enable_pe_control signal is used instead of the enable_pe_array signal as the PE controller is always enabled one
        // clock cycle earlier, giving the input memory one clock cycle to prepare the data
        .read_enable(is_input_layer & enable_pe_control),
        .address(in_activation_control_address[InputAddressWidth-1:0]),

        .write_new(write_new),
        .read_sync(read_sync),
        .code_is_input(code_is_input),

        .spi_address(spi_address),
        .input_spi_data_in(MOSI_data),
        .input_spi_data_out(input_spi_data_out),

        .in_idle(in_idle),
        .running(running),

        .fill_first(fill_input_memory),
        .require_single_chunk(require_single_chunk),
        .use_subchunks(use_subchunks),
        .load_inputs_from_activation_memory(load_inputs_from_activation_memory),

        .in_first_block(in_first_block),
        .in_last_block_of_input_layer(in_last_block_of_input_layer),
        .is_layer_residual(is_layer_residual),
        .input_blocks(blocks_per_layer[0]),
        .input_blocks_times_kernel_size(input_blocks_times_kernel_size),

        .data_available(intercepted_data_available),
        .data_required (data_required),

        .ready(is_input_memory_ready),

        .data_in (data_in),
        .data_out(input_data_in)
    );

    // PE array control ----------------------------------------------------------------------------

    assign is_input_layer = is_input_layer_pe_control & ~processing_few_shot;  // TODO: this was a quick fix

    pe_array_control #(
        .ACTIVATION_ADDRESS_WIDTH(ACTIVATION_ADDRESS_WIDTH),
        .WEIGHT_ADDRESS_WIDTH(WEIGHT_ADDRESS_WIDTH),
        .BIAS_ADDRESS_WIDTH(BIAS_ADDRESS_WIDTH),
        .MAX_NUM_LAYERS(MAX_NUM_LAYERS),
        .BLOCKS_WIDTH(BLOCKS_WIDTH),
        .KERNEL_WIDTH(KERNEL_WIDTH),
        .BLOCKS_KERNEL_WIDTH(BLOCKS_KERNEL_WIDTH),
        .CUMSUM_WIDTH(CumsumWidth)
    ) pe_array_control_inst (
        .clk(clk),
        .rst(rst_sync | restart_pe_array_control),

        // Inputs
        .enable (enable_pe_control),

        // Configuration parameters
        .num_conv_layers(num_conv_layers),
        .num_conv_and_linear_layers(num_conv_and_linear_layers_few_shot_corrected),
        .kernel_size_per_layer(kernel_size_per_layer),
        .blocks_per_layer(blocks_per_layer),
        .blocks_per_layer_times_kernel_size_cumsum(blocks_per_layer_times_kernel_size_cumsum),
        .few_shot_next_layer_blocks(blocks_per_layer_few_shot_corrected),

        .force_downsample(force_downsample),
        .load_inputs_from_activation_memory(load_inputs_from_activation_memory),
        .use_few_shot_next_layer_blocks(use_few_shot_next_layer_blocks),

        .in_context_learning(in_context_learning),
        .num_conv_and_linear_layers_icl_embedder(num_conv_and_linear_layers),
        .input_blocks_times_kernel_size_icl_head(input_blocks_times_kernel_size_icl_head),

        .processed_all_input_blocks_into_current_block_out(processed_all_input_blocks_into_current_block_out),

        // Outputs
        .enable_pe_array(enable_pe_array),

        .apply_identity(apply_identity),
        .dont_load_weights(dont_load_weights),
        .load_scale_and_bias(load_scale_and_bias),
        .use_scale_and_bias(use_scale_and_bias),
        .dont_wait_for_linear_layer(dont_wait_for_linear_layer),
        .apply_in_scale(apply_in_scale),
        .output_ready(output_ready),
        .layer(layer),
        .delayed_layer(delayed_layer),
        .in_last_cycle_of_address_generation(in_last_cycle_of_address_generation),

        .is_input_layer(is_input_layer_pe_control),
        .feed_input_data_to_pe_array(feed_input_data_to_pe_array),
        .is_layer_residual(is_layer_residual),
        .is_output_layer(is_output_layer),
        .in_cycle_before_sending_embedding_out(in_cycle_before_sending_embedding_out),
        .in_first_block(in_first_block),
        .in_last_block_of_input_layer(in_last_block_of_input_layer),

        // Addresses for the memories and the PE array
        .in_activation_address(in_activation_address),
        .weight_address(weight_address_pe_control),
        .bias_address(bias_address_pe_control),
        .out_activation_address(out_activation_address)
    );

    // PE array ----------------------------------------------------------------------------------

    assign in_activation_data = feed_input_data_to_pe_array & ~processing_few_shot ? input_data_in : activation_data_out;

    convert_1d_to_2d_array #(
        .BIT_WIDTH(ACTIVATION_BIT_WIDTH),
        .COLS(PE_ROWS)
    ) convert_in_activation_data_vector (
        .in (in_activation_data),
        .out(in_activation_data_vector)
    );

    convert_1d_to_3d_sub_array #(
        .BIT_WIDTH(WEIGHT_BIT_WIDTH),
        .ROWS(PE_ROWS),
        .COLS(PE_COLS),
        .SUB_ROWS(SUBSECTION_SIZE),
        .SUB_COLS(SUBSECTION_SIZE)
    ) convert_weight_data_out_vector (
        .in (weight_data_out),
        .out(weight_data_out_vector)
    );

    convert_1d_to_2d_array #(
        .BIT_WIDTH(BIAS_BIT_WIDTH),
        .COLS(PE_COLS)
    ) convert_bias_data_out_vector (
        // Input bias = 0 when we are in few-shot mode as we are calculating L2 distance which does not require bias
        .in (processing_few_shot ? 0 : bias_data_out),
        .out(biases)
    );

    convert_2d_to_1d_array #(
        .BIT_WIDTH(ACTIVATION_BIT_WIDTH),
        .COLS(PE_COLS)
    ) convert_out_activation_data_vector (
        .in (out_activation_data_vector),
        .out(activation_control_data_in)
    );

    // enable_pe_array_during_few_shot is only high when we are in few-shot mode and we are processing the few-shot samples
    assign enable_pe_array_few_shot_corrected = enable_pe_array_during_few_shot | enable_pe_array;
    assign scale_and_bias_ready = processing_few_shot ? scale_and_bias_ready_few_shot : use_scale_and_bias;

    assign shift_few_shot_data_right = few_shot_scale[FewShotScaleWidth-1];
    assign left_shift_few_shot_scale = shift_few_shot_data_right ? 0 : few_shot_scale;
    assign right_shift_few_shot_scale = shift_few_shot_data_right ? few_shot_scale : 0;

    assign {in_scale, out_scale} = scale_and_residual_scale_per_layer[delayed_layer];
    assign in_scale_few_shot_corrected = enable_pe_array_during_few_shot ? left_shift_few_shot_scale : in_scale;

    pe_array #(
        .ROWS(PE_ROWS),
        .COLS(PE_COLS),
        .ACTIVATION_BIT_WIDTH(ACTIVATION_BIT_WIDTH),
        .WEIGHT_BIT_WIDTH(WEIGHT_BIT_WIDTH),
        .BIAS_BIT_WIDTH(BIAS_BIT_WIDTH),
        .SCALE_BIT_WIDTH(SCALE_BIT_WIDTH),
        .ACCUMULATION_BIT_WIDTH(ACCUMULATION_BIT_WIDTH),
        .SUBSECTION_SIZE(SUBSECTION_SIZE)
    ) pe_array_inst (
        .clk(clk),
        .enable(enable_pe_array_few_shot_corrected),

        .apply_identity(apply_identity | enable_pe_array_during_few_shot),
        .apply_in_scale(apply_in_scale | enable_pe_array_during_few_shot),
        .use_subsection(in_4x4_mode),
        .apply_bias(scale_and_bias_ready),

        .in(in_activation_data_vector),
        .weights(weight_data_out_vector),
        .biases(biases),
        .out_scale(out_scale),
        .in_scale(in_scale_few_shot_corrected),

        .out(out_activation_data_vector),
        .col_accumulator(col_accumulator)
    );

    // Few-shot control ---------------------------------------------------------------------------

    // When doing continual learning, the layer before the last layer is the embedding layer
    // Note: this is not a timing-critical path as all inputs into this wire are from the SPI configuration registers
    assign embedder_layer_index = continued_learning ? num_conv_and_linear_layers - 1 : num_conv_and_linear_layers;

    double_flop_synchronizer #(
        .AT_POSEDGE_RST(0)
    ) double_flop_synchronizer_is_new_task_sync (
        .clk(clk),
        .rst(rst_sync),

        .enable(1'b1),

        .in (is_new_task),
        .out(is_new_task_sync)
    );

    few_shot_learning_control #(
        .SHOT_BIT_WIDTH(SHOT_BIT_WIDTH),
        .STATE_BIT_WIDTH(StateBitWidth),
        .ACTIVATION_ADDRESS_WIDTH(ACTIVATION_ADDRESS_WIDTH),
        .WEIGHT_ADDRESS_WIDTH(WEIGHT_ADDRESS_WIDTH),
        .BIAS_ADDRESS_WIDTH(BIAS_ADDRESS_WIDTH),
        .BLOCKS_WIDTH(BLOCKS_WIDTH),
        .PE_COLS(PE_COLS),
        .SUBSECTION_SIZE(SUBSECTION_SIZE),
        .LAYER_WIDTH(LAYER_WIDTH),
        .CUMSUM_WIDTH(CumsumWidth),
        .WAYS_BIT_WIDTH(ArgmaxIndexWidth)
    ) few_shot_learning_control_inst (
        .clk(clk),
        .rst(rst_sync),

        .state(state),

        .num_conv_and_linear_layers(num_conv_and_linear_layers),
        .num_conv_and_linear_layers_full_icl_net(num_conv_and_linear_layers_full_icl_net),
        .cumsum_conv_layers(blocks_per_layer_times_kernel_size_cumsum[embedder_layer_index]),
        .embedding_layer_blocks(blocks_per_layer[embedder_layer_index]),
        .linear_layer_blocks(blocks_per_layer[num_conv_and_linear_layers]),

        .is_new_task_sync(is_new_task_sync),
        .classification(classification),
        .continued_learning(continued_learning),
        .in_context_learning(in_context_learning),
        .skip_sending(skip_sending),
        .toggle_processing_new(toggle_processing_new),
        .is_output_layer(is_output_layer),
        .use_subsection(in_4x4_mode),

        .shots(shots),
        .max_weight_address(max_weight_address),
        .max_bias_address(max_bias_address),

        .out_activation_address(out_activation_address),
        .out_activation_address_few_shot_corrected(out_activation_control_address),

        .in_activation_address(in_activation_address),
        .in_activation_address_few_shot_corrected(in_activation_control_address),

        .weight_address_few_shot_corrected(weight_address_few_shot_corrected),
        .bias_address_few_shot_corrected  (bias_address_few_shot_corrected),

        .processing_new_task_samples(processing_new_task_samples),
        .enable_activation_memory_for_few_shot(enable_activation_memory_for_few_shot),
        .enable_weight_memory_for_few_shot(enable_weight_memory_for_few_shot),
        .enable_pe_array(enable_pe_array_during_few_shot),
        .start_few_shot_processing(start_few_shot_processing),
        .done_few_shot_processing(done_few_shot_processing),
        .scale_and_bias_ready_few_shot(scale_and_bias_ready_few_shot),
        .enable_bias_memory_for_few_shot(enable_bias_memory_for_few_shot),
        .configured_for_few_shot_processing(configured_for_few_shot_processing),
        .use_few_shot_next_layer_blocks(use_few_shot_next_layer_blocks),
        .ways_received(ways_received),
        .shots_received(shots_received),
        .weight_bias_offset(weight_bias_offset),

        .few_shot_shift_required(few_shot_shift_required),
        .num_conv_and_linear_layers_few_shot_corrected(num_conv_and_linear_layers_few_shot_corrected),
        .blocks_per_layer_few_shot_corrected(blocks_per_layer_few_shot_corrected)
    );

    take_lsbs_of_2d_array #(
        .IN_BIT_WIDTH(ACCUMULATION_BIT_WIDTH),
        .OUT_BIT_WIDTH(FLog2Width),
        .SCALE_BIT_WIDTH(RightShiftFewShotScaleWidth),
        .COLS(PE_COLS)
    ) take_lsbs_of_col_accumulator (
        .in (col_accumulator),
        .scale(right_shift_few_shot_scale),
        .out(col_accumulator_lsbs)
    );

    wide_flog2 #(
        .COLS(PE_COLS),
        .BIT_WIDTH(FLog2Width)
    ) wide_flog2_inst (
        .in (col_accumulator_lsbs),
        .out(new_few_shot_weights)
    );

    convert_2d_to_1d_array_size #(
        .BIT_WIDTH(LeftShiftFewShotScaleWidth),
        .OUT_BIT_WIDTH(WEIGHT_BIT_WIDTH),
        .COLS(PE_COLS)
    ) convert_new_few_shot_weights (
        .in (new_few_shot_weights),
        .out(new_few_shot_weights_flattened)
    );

    // Squared-sum accumulator --------------------------------------------------------------------

    squared_log2_sum_accumulator #(
        .COLS(PE_COLS),
        .SUB_COLS(SUBSECTION_SIZE),
        .BIT_WIDTH(FLog2Width),
        .LEFT_SHIFT_FEW_SHOT_SCALE_WIDTH(LeftShiftFewShotScaleWidth),
        .RIGHT_SHIFT_FEW_SHOT_SCALE_WIDTH(RightShiftFewShotScaleWidth),
        .K_SHOT_SCALE_WIDTH(KShotDivisionScaleWidth),
        .BIAS_BIT_WIDTH(BIAS_BIT_WIDTH),
        .ACCUMULATION_BIT_WIDTH(FEW_SHOT_ACCUMULATION_BIT_WIDTH)
    ) squared_log2_sum_accumulator_inst (
        .clk(clk),
        .rst(rst_sync | in_idle),

        .enable(enable_weight_memory_for_few_shot & use_l2_for_few_shot),
        .in_4x4_mode(in_4x4_mode),

        .shift_few_shot_data_right(shift_few_shot_data_right),
        .left_shift_few_shot_scale(left_shift_few_shot_scale),
        .right_shift_few_shot_scale(right_shift_few_shot_scale),
        .k_shot_division_scale(k_shot_division_scale),

        .in (new_few_shot_weights),
        .out(new_few_shot_bias)
    );

    // Serial-parallel argmax ----------------------------------------------------------------------

    convert_2d_to_1d_array #(
        .BIT_WIDTH(ACCUMULATION_BIT_WIDTH),
        .COLS(PE_COLS)
    ) convert_col_accumulator_to_vector (
        .in (col_accumulator),
        .out(col_accumulator_vector_argmax)
    );

    convert_1d_to_2d_array #(
        .BIT_WIDTH(ACCUMULATION_BIT_WIDTH),
        .COLS(PE_COLS)
    ) convert_col_accumulator_vector (
        .in (enable_argmax ? col_accumulator_vector_argmax : 0),
        .out(data_gated_col_accumulator_vector_argmax)
    );

    // Disable lint for .max() output of module since we do not care about its value
    serial_parallel_argmax #(
        .WIDTH(ACCUMULATION_BIT_WIDTH),
        .N(PE_COLS),
        .INPUT_DATA_SHIFT(ARGMAX_INPUTS_SHIFT),
        .SERIAL_ARGMAX_WIDTH(ArgmaxIndexWidth),
        .LEFT_SHIFT_WIDTH(LeftShiftWidth)
    ) u_serial_parallel_argmax (
        .clk(clk),
        .rst(rst_sync | restart_argmax),

        .enable(enable_argmax),

        .inputs_left_shift(0),

        .data(data_gated_col_accumulator_vector_argmax),

        .argmax(argmax),
        .max(max_out_value)
    );

    // High-speed out bus -------------------------------------------------------------------------

    assign data_to_output = (classification ? {max_out_value, argmax} >> (sent_counter*HIGH_SPEED_OUT_PINS) : activation_control_data_in[HIGH_SPEED_OUT_PINS*sent_counter+:HIGH_SPEED_OUT_PINS]);
    assign num_sends = classification ? (send_all_argmax_chunks ? (ArgmaxIndexWidth + HIGH_SPEED_OUT_PINS + ACCUMULATION_BIT_WIDTH - 1) / HIGH_SPEED_OUT_PINS : 1) : (in_4x4_mode ? SUBSECTION_SIZE : PE_COLS) * ACTIVATION_BIT_WIDTH / HIGH_SPEED_OUT_PINS;

    // When argmax is enabled but the PE array not anymore, it means that in the next clock cycle
    // the final output of the argmax is ready.
    assign start_sending = (enable_argmax & ~enable_pe_array) | (is_output_layer & output_ready & ~classification & ~processing_new_task_samples);

    high_speed_out_bus #(
        .HIGH_SPEED_OUT_PINS(ArgmaxIndexWidth),
        .SENT_COUNTER_BIT_WIDTH(SentCounterBitWidth)
    ) high_speed_out_bus_inst (
        .clk(clk),
        .rst(rst_sync),

        .in_idle(in_idle),
        .sending(sending),
        .will_stop_sending(next_state != `SENDING),

        .data_ready_for_sending(1'b1),

        .num_sends(num_sends),

        .request(out_request),
        .acknowledge(in_acknowledge),

        .in(data_to_output),

        .out(data_out),
        .sent_counter(sent_counter),

        .done_sending(done_sending)
    );

    // Global SRAM power-down controller ----------------------------------------------------------

    always @(posedge clk) begin
        // TODO: not sure if rst_sync signal is needed
        if (rst_sync | in_idle) begin
            keep_memories_powered_up <= 1'b0;
        end else if (restart_input_memory) begin
            // If we are going to restart the input memory (in the last cycle of creating the embeddings) for the current
            // task, then in the next cycle the input memory will not be ready anymore. This will put the
            // memories in power down mode while they still have to written to for few-shot processing. Therefore,
            // as soon as the input memory will restart, we set this keep_memories_powered_up flag high to make sure that
            // the memories dont go into power down mode.
            keep_memories_powered_up <= 1'b1;
        end
    end

    // The syncing below is necessary, otherwise when putting, for example,
    // power_down_srams_in_standby high, if it doesnt go high on exactly the right
    // moment, a setup violation will take place and, among others, the 
    // global_power_down wire will go to 'x'.
    double_flop_synchronizer #(
        .AT_POSEDGE_RST(0)
    ) double_flop_synchronizer_power_down_memories_while_running_sync (
        .clk(clk),
        .rst(rst_sync),

        .enable(in_idle),

        .in (power_down_memories_while_running),
        .out(power_down_memories_while_running_sync)
    );

    double_flop_synchronizer #(
        .AT_POSEDGE_RST(0)
    ) double_flop_synchronizer_power_down_srams_in_standby_sync (
        .clk(clk),
        .rst(rst_sync),

        .enable(in_idle),

        .in (power_down_srams_in_standby),
        .out(power_down_srams_in_standby_sync)
    );

    global_sram_power_down_controller #(
        .WAIT_CYCLES_WIDTH(WAIT_CYCLES_WIDTH)
    ) global_sram_power_down_controller_inst (
        .clk(clk),
        .rst(rst_sync),

        .wake_up_delay(wake_up_delay),
        .power_up_delay(power_up_delay),

        .should_be_on  (is_input_memory_ready | keep_memories_powered_up | (in_idle & ~power_down_srams_in_standby_sync)),
        .allow_power_down((running & power_down_memories_while_running_sync) | (in_idle & power_down_srams_in_standby_sync)),

        .memories_ready(memories_ready),
        .in_power_down (global_power_down)
    );

    // Weight memory ------------------------------------------------------------------------------

    // The weight memory needs to be active in the first cycle when the PE array control is enabled,
    // as its outputs are the the 0th step addresses, but also should remain active one cycle longer
    // than the PE array control, since when the enable signal goes low, also the last address of the
    // controller will be outputed, which we still need to process.
    assign weight_control_chip_select = (enable_pe_control & ~dont_load_weights) | enable_weight_memory_for_few_shot;
    assign weight_control_write_enable = enable_weight_memory_for_few_shot;

    // By starting with {WeightWordBitWidth{1'b1}}, we make sure to write all bias bits to make sure they are all 0
    assign subsection_weight_mask = {SUBSECTION_ACTIVATION_WORD_BIT_WIDTH{1'b1}} << (few_shot_shift_required * SUBSECTION_ACTIVATION_WORD_BIT_WIDTH);
    assign rest_weight_mask = {OnesLengthWeight{1'b1}} << (few_shot_shift_required * OnesLengthWeight);

    assign subsection_weight_data = new_few_shot_weights_flattened[SUBSECTION_ACTIVATION_WORD_BIT_WIDTH-1:0] << (few_shot_shift_required * SUBSECTION_ACTIVATION_WORD_BIT_WIDTH);
    assign rest_weight_data = new_few_shot_weights_flattened[ACTIVATION_WORD_BIT_WIDTH-1:SUBSECTION_ACTIVATION_WORD_BIT_WIDTH] << (few_shot_shift_required * OnesLengthWeight);

    assign weight_control_mask = enable_weight_memory_for_few_shot ? (few_shot_shift_required == 0 ? {WeightWordBitWidth{1'b1}} : {rest_weight_mask, subsection_weight_mask}) : 0;
    assign weight_control_data_in = {rest_weight_data, subsection_weight_data};
    assign weight_control_address = enable_weight_memory_for_few_shot ? weight_address_few_shot_corrected : weight_address_pe_control;

    managed_weight_memory #(
        .WEIGHT_WORD_BIT_WIDTH(WeightWordBitWidth),
        .WEIGHT_ROWS(WEIGHT_ROWS),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) managed_weight_memory_inst (
        .clk(clk),
        .rst(rst_sync),

        .write_new(write_new),
        .read_sync(read_sync),
        .code_is_weight(code_is_weight),

        .in_4x4_mode(in_4x4_mode),

        .spi_address(spi_address),
        .weights_spi_data_in(MOSI_data),
        .weight_spi_data_out(weight_spi_data_out),

        .weight_control_chip_select(weight_control_chip_select),
        .weight_control_write_enable(weight_control_write_enable),
        .global_power_down(global_power_down),
        .weight_control_address(weight_control_address),
        .weight_control_data_in(weight_control_data_in),
        .weight_control_mask(weight_control_mask),

        .weight_data_out(weight_data_out)
    );

    // Bias memory --------------------------------------------------------------------------------

    // Similarly to the read_enable from the input_memory: the load_scale_and_bias signal is already high
    // a long time before we start processing, so we only really want to read from the bias memory when
    // the PE controller is enabled.
    assign bias_control_chip_select = (load_scale_and_bias && enable_pe_control) | enable_bias_memory_for_few_shot;
    assign bias_control_write_enable = enable_bias_memory_for_few_shot;

    // By starting with {BiasWordBitWidth{1'b1}}, we make sure to write all bias bits to make sure they are all 1
    assign bias_control_mask = enable_bias_memory_for_few_shot ? (few_shot_shift_required == 0 ? {BiasWordBitWidth{1'b1}} : ({BIAS_BIT_WIDTH{1'b1}} << (few_shot_shift_required * BIAS_BIT_WIDTH))) : 0;

    assign bias_control_data_in = enable_bias_memory_for_few_shot ? (few_shot_shift_required == 0 ? {{NumberOfMostNegativeBiases{{1'b1}, {NumberOfZeroesInMostNegativeBias{1'b0}}}}, new_few_shot_bias} : (new_few_shot_bias << (few_shot_shift_required * BIAS_BIT_WIDTH))) : 0;
    assign bias_control_address = enable_bias_memory_for_few_shot ? bias_address_few_shot_corrected : bias_address_pe_control;

    managed_bias_memory #(
        .BIAS_WORD_BIT_WIDTH(BiasWordBitWidth),
        .BIAS_ROWS(BIAS_ROWS),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) managed_bias_memory_inst (
        .clk(clk),

        .write_new(write_new),
        .read_sync(read_sync),
        .code_is_bias(code_is_bias),

        .power_down_small_bias(power_down_small_bias),

        .spi_address(spi_address),
        .bias_spi_data_in(MOSI_data),
        .bias_spi_data_out(bias_spi_data_out),

        .bias_control_chip_select(bias_control_chip_select),
        .bias_control_write_enable(bias_control_write_enable),
        .global_power_down(global_power_down),
        .bias_control_address(bias_control_address),
        .bias_control_data_in(bias_control_data_in),
        .bias_control_mask(bias_control_mask),

        .bias_data_out(bias_data_out)
    );

    // Activation memory --------------------------------------------------------------------------

    // Only chip select the activation memory when we want to load activations from this memory
    // (condition 1) or when we want to write to this memory (condition 2).
    assign activation_control_read_enable = ((~is_input_layer) & enable_pe_control) | enable_activation_memory_for_few_shot;
    assign activation_control_write_enable = output_ready;
    assign activation_control_mask = '1;

    managed_activation_memory #(
        .ACTIVATION_WORD_BIT_WIDTH(ACTIVATION_WORD_BIT_WIDTH),
        .ACTIVATION_ROWS(ACTIVATION_ROWS),
        .START_ADDRESS_BIT_WIDTH(START_ADDRESS_BIT_WIDTH),
        .MESSAGE_BIT_WIDTH(MESSAGE_BIT_WIDTH)
    ) managed_activation_memory_inst (
        .clk(clk),

        .write_new(write_new),
        .read_sync(read_sync),
        .code_is_activation(code_is_activation),

        .spi_address(spi_address),
        .activations_spi_data_in(MOSI_data),
        .activation_spi_data_out(activation_spi_data_out),

        .activation_control_read_enable(activation_control_read_enable),
        .activation_control_write_enable(activation_control_write_enable),
        .global_power_down(global_power_down),
        .activation_control_address_read(in_activation_control_address),
        .activation_control_address_write(out_activation_control_address),
        .activation_control_data_in(activation_control_data_in),
        .activation_control_mask(activation_control_mask),

        .activation_data_out(activation_data_out)
    );

endmodule
