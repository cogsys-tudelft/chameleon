module pe_array_control
/**
  *  This module controls the PE array. It is responsible for generating the addresses for the
  *  activation, weight and bias memories. These addresses are generating assuming that the
  *  to be processed network is a TCN and that the processing takes place with batch size = 1.
  *  as a reference.
  *
  *  Parameters:
  *  - ACTIVATION_ADDRESS_WIDTH: Width of the activation memory address
  *  - WEIGHT_ADDRESS_WIDTH: Width of the weight memory address
  *  - BIAS_ADDRESS_WIDTH: Width of the bias memory address
  *  - MAX_NUM_LAYERS: Maximum number of layers that can be processed (one TCN layer is two layers, one linear layer is one layer)
  *  - BLOCKS_WIDTH: Width of the blocks_per_layer input
  *  - KERNEL_WIDTH: Width of the kernel_size_per_layer input
  *  - BLOCKS_KERNEL_WIDTH: Width of the blocks_per_layer_times_kernel_size input
  *
  *  Inputs:
  *  - clk: Clock
  *  - rst: Reset
  *
  *  Outputs:
  *   - argmax: Index of the maximum value
  */
#(
    parameter integer ACTIVATION_ADDRESS_WIDTH = 8,
    parameter integer WEIGHT_ADDRESS_WIDTH = 8,
    parameter integer BIAS_ADDRESS_WIDTH = 8,
    parameter integer MAX_NUM_LAYERS = 16,
    parameter integer BLOCKS_WIDTH = 4,
    parameter integer KERNEL_WIDTH = 4,
    parameter integer BLOCKS_KERNEL_WIDTH = 8,
    parameter integer CUMSUM_WIDTH = 8,

    localparam integer LayerWidth  = $clog2(MAX_NUM_LAYERS)
) (
    input clk,
    input rst,

    input enable,

    input [LayerWidth-1:0] num_conv_layers,
    input [LayerWidth-1:0] num_conv_and_linear_layers,
    input [KERNEL_WIDTH-1:0] kernel_size_per_layer[MAX_NUM_LAYERS:0],
    input [BLOCKS_WIDTH-1:0] blocks_per_layer[MAX_NUM_LAYERS:0],  // Add one extra entry to account for input layer
    input [CUMSUM_WIDTH-1:0] blocks_per_layer_times_kernel_size_cumsum[MAX_NUM_LAYERS:0],  // Because of the special logic for layer 0 and 1, can leave the first two elements out here
    input [BLOCKS_WIDTH-1:0] few_shot_next_layer_blocks,

    input force_downsample,
    input load_inputs_from_activation_memory,
    input use_few_shot_next_layer_blocks,
    input in_context_learning,
    input [LayerWidth-1:0]num_conv_and_linear_layers_icl_embedder,
    input [BLOCKS_KERNEL_WIDTH-1:0] input_blocks_times_kernel_size_icl_head,

    output enable_pe_array,

    output apply_identity,
    output dont_load_weights,
    output reg load_scale_and_bias,
    output use_scale_and_bias,
    output reg dont_wait_for_linear_layer,
    output apply_in_scale,
    output output_ready,
    output reg [LayerWidth-1:0] layer,
    output [LayerWidth-1:0] delayed_layer,

    output processed_all_input_blocks_into_current_block_out,

    output is_output_layer,
    output is_layer_residual,
    output is_input_layer,
    output feed_input_data_to_pe_array,
    output in_cycle_before_sending_embedding_out,
    output in_last_cycle_of_address_generation,

    output in_first_block,
    output in_last_block_of_input_layer,

    output [ACTIVATION_ADDRESS_WIDTH-1:0] in_activation_address,
    output reg [WEIGHT_ADDRESS_WIDTH-1:0] weight_address,
    output reg [BIAS_ADDRESS_WIDTH-1:0] bias_address,
    output [ACTIVATION_ADDRESS_WIDTH-1:0] out_activation_address
);

    // Running variables --------------------------------------------------------------------------

    // All these registers have size ACTIVATION_ADDRESS_WIDTH-1:0 as anyway the this is the maximum
    // input and output address we can generate to access the memory
    reg [BLOCKS_KERNEL_WIDTH-1:0] input_address_per_layer[MAX_NUM_LAYERS];
    reg [BLOCKS_KERNEL_WIDTH-1:0] prev_input_address;
    reg [KERNEL_WIDTH-1:0] steps_in_kernel_counter[MAX_NUM_LAYERS];
    reg [BLOCKS_KERNEL_WIDTH-1:0] out_address_per_layer [MAX_NUM_LAYERS]; // TODO: maybe can be 1 bit less wide // TODO! think if it's needed to have this. Maybe we can compute it from steps in kernel counter etc.

    reg [BLOCKS_WIDTH-1:0] residual_steps;
    reg [BLOCKS_WIDTH-1:0] out_blocks_completed_counter;
    reg [BLOCKS_KERNEL_WIDTH-1:0] steps_in_input_layer;

    reg processing_residual_part;

    // Temporary variables for always-on logic ----------------------------------------------------

    wire [ACTIVATION_ADDRESS_WIDTH-1:0] local_input_address;

    wire processing_linear_layer;
    wire has_one_or_more_linear_layers_at_the_end;
    wire is_odd_layer_number;
    wire apply_identity_internal;
    wire current_layer_blocks_same_as_previous_layer_blocks;

    wire processed_all_input_blocks_into_current_block;

    wire is_output_layer_internal;

    wire not_waiting_in_output_layer;
    wire is_next_layer_residual;
    wire in_last_step;
    wire is_true_input_layer;

    wire is_layer_zero;
    wire is_layer_one;

    wire [2-1:0] re_eval_count;  // Only needs to store 1 and 2, so 2 bits is enough
    wire [BLOCKS_WIDTH+1-1:0] shift_times_blocks_per_layer;

    wire [CUMSUM_WIDTH-1:0] block_sum_in;

    wire [ACTIVATION_ADDRESS_WIDTH-1:0] earlier_out_activation_address;

    wire [BLOCKS_KERNEL_WIDTH-1:0] blocks_times_kernel_for_layer;
    wire [BLOCKS_KERNEL_WIDTH-1:0] blocks_times_kernel_for_next_layer;

    // Always-on logic ----------------------------------------------------------------------------

    assign is_layer_zero = (layer == 0);
    assign is_layer_one = (layer == 1);

    assign in_first_block = out_blocks_completed_counter == 0;
    assign in_last_block_of_input_layer = out_blocks_completed_counter == blocks_per_layer[1] && is_layer_zero;
    assign current_layer_blocks_same_as_previous_layer_blocks = blocks_per_layer[layer-1] == blocks_per_layer[layer+1];
    assign apply_identity_internal = ~force_downsample & processing_residual_part & current_layer_blocks_same_as_previous_layer_blocks;
    assign dont_load_weights = apply_identity_internal;

    assign processing_linear_layer = layer + 1 > num_conv_layers;
    assign has_one_or_more_linear_layers_at_the_end = num_conv_and_linear_layers > num_conv_layers;
    assign is_odd_layer_number = layer % 2 == 1;

    assign processed_all_input_blocks_into_current_block = steps_in_input_layer == blocks_times_kernel_for_layer;

    assign is_true_input_layer = is_layer_zero | (processing_residual_part & is_layer_one);
    assign is_input_layer = is_true_input_layer & ~load_inputs_from_activation_memory;

    // Also cover for the special case where there is only one layer, and the input and output layer are the same
    // this line makes sure that when there is only one layer, Chameleon does not get immediately triggered
    // after one high pe_control_enable cycle to start sending outputs since 'it is in the output layer',
    // while actually the output block is not ready yet.
    assign is_output_layer_internal = num_conv_and_linear_layers == 1 ? processed_all_input_blocks_into_current_block & enable : layer + 1 == num_conv_and_linear_layers;

    assign not_waiting_in_output_layer = is_output_layer_internal & dont_wait_for_linear_layer & ~processing_residual_part;

    assign is_layer_residual = processing_linear_layer ? 0 : is_odd_layer_number;
    assign is_next_layer_residual = !is_odd_layer_number & !processing_linear_layer;
    assign re_eval_count = processing_linear_layer ? 1 : (is_odd_layer_number ? 1 : 2);
    assign shift_times_blocks_per_layer = processing_linear_layer ? 0 : (is_odd_layer_number ? ((blocks_per_layer[layer] + 1) * 2) : blocks_per_layer[layer] + 1);

    // If layer is one, then layer - 1 gives 0 which is always zero in blocks_per_layer_times_kernel_size_cumsum due to the way we construct it.
    // That's why we initially check if we are in layer 1 and then set the block_sum_in to 0 to avoid the 0 index.
    assign block_sum_in = blocks_per_layer_times_kernel_size_cumsum[processing_residual_part ? layer-1 : layer];

    assign local_input_address = processing_residual_part ? prev_input_address + residual_steps - blocks_per_layer[layer-1] : input_address_per_layer[layer];

    assign in_activation_address = local_input_address + block_sum_in;
    assign earlier_out_activation_address = out_address_per_layer[layer] + blocks_per_layer_times_kernel_size_cumsum[layer+1];

    // TODO: could replace this long part with the maximum weight index == weight_index
    assign in_last_cycle_of_address_generation = not_waiting_in_output_layer & (out_blocks_completed_counter == (use_few_shot_next_layer_blocks ? few_shot_next_layer_blocks : blocks_per_layer[num_conv_and_linear_layers]) && processed_all_input_blocks_into_current_block);
    assign in_cycle_before_sending_embedding_out = not_waiting_in_output_layer & processed_all_input_blocks_into_current_block;

    assign blocks_times_kernel_for_layer = in_context_learning && layer == num_conv_and_linear_layers_icl_embedder ? input_blocks_times_kernel_size_icl_head : ((blocks_per_layer[layer] + 1) * kernel_size_per_layer[layer] - 1);
    assign blocks_times_kernel_for_next_layer = (blocks_per_layer[layer+1] + 1) * kernel_size_per_layer[layer+1] - 1;

    // Delays for PE array ------------------------------------------------------------------------

    double_delay_register double_delay_register_is_output_layer (
        .clk(clk),
        .rst(rst),
        .in(is_output_layer_internal),
        .in_delayed(is_output_layer)
    );

    delay_register delay_register_processing_residual_part (
        .clk(clk),
        .rst(rst),
        .in(processing_residual_part),
        .in_delayed(apply_in_scale)
    );

    delay_register delay_register_is_input_layer (
        .clk(clk),
        .rst(rst),
        .in(is_input_layer),
        .in_delayed(feed_input_data_to_pe_array)
    );

    double_delay_register double_delay_register_is_output_layer2 (
        .clk(clk),
        .rst(rst),
        .in(processed_all_input_blocks_into_current_block),
        .in_delayed(processed_all_input_blocks_into_current_block_out)
    );

    // Only enable the PE array 1 cycle after this controller, as 1 cycle is required to load all the data
    // from the various memories for the PE array to use
    delay_register delay_register_enable_pe_control (
        .clk(clk),
        .rst(rst),
        .in(enable),
        .in_delayed(enable_pe_array)
    );

    delay_register delay_register_output_ready (
        .clk(clk),
        .rst(rst),
        .in(load_scale_and_bias && enable_pe_array),  // TODO: explain!!
        .in_delayed(output_ready)
    );

    // The PE array can only use the scale and bias 1 cycle after it has been loaded from the SRAM
    delay_register delay_register_load_scale_and_bias (
        .clk(clk),
        .rst(rst),
        .in(load_scale_and_bias),
        .in_delayed(use_scale_and_bias)
    );

    // Need to delay the output address twice, as it takes 1 cycle to load the input data and
    // one more cycle to calculate the output data
    wide_double_delay_register #(
        .WIDTH(ACTIVATION_ADDRESS_WIDTH)
    ) wide_double_delay_register_out_activation_address (
        .clk(clk),
        .rst(rst),
        .in(earlier_out_activation_address),
        .in_delayed(out_activation_address)
    );

    // Need a delay one cycle because in the first cycle we turn off the weight memory while in the second cycle we enable the multipliers
    delay_register delay_register_apply_identity (
        .clk(clk),
        .rst(rst),
        .in(apply_identity_internal),
        .in_delayed(apply_identity)
    );

    wide_delay_register #(
        .WIDTH(LayerWidth)
    ) wide_delay_register_layer (
        .clk(clk),
        .rst(rst),
        .in(layer),
        .in_delayed(delayed_layer)
    );

    // Finite-state logic -------------------------------------------------------------------------

    always @(posedge clk) begin
        if (rst) begin
            weight_address <= 0;
            bias_address <= 0;

            for (integer i = 0; i < MAX_NUM_LAYERS; i = i + 1) begin
                input_address_per_layer[i] <= 0;
                steps_in_kernel_counter[i] <= 0;
                out_address_per_layer[i] <= 0;
            end

            // No need to set prev_input_address to 0 as it is always first written to before
            // being read from. It is also never incremented or something like that.

            layer <= 0;
            residual_steps <= 0;
            out_blocks_completed_counter <= 0;
            steps_in_input_layer <= 0;

            processing_residual_part <= 1'b0;

            load_scale_and_bias <= 1'b1;
            dont_wait_for_linear_layer <= 1'b1;
        end else begin
            // Dont wait for linear layer is "Dont" as then we dont have to do ! for the enable_pe_control,
            // since it already is a not-signal. This signal is generally needed, due to the fact that when
            // switching from the TCN to the linear layer, the last output of the TCN is the input of the
            // linear layer. This means that we need to wait for the TCN to finish before we can start the
            // linear layer. The same problem occurs when going from one linear layer with only 1 block to
            // another linear layer.
            if (dont_wait_for_linear_layer == 1'b0 && (output_ready == 1'b1 || blocks_per_layer[layer] == 2 || processing_residual_part)) begin
                dont_wait_for_linear_layer <= 1'b1;
            end

            if (enable == 1'b1) begin
                if (~apply_identity_internal) begin
                    weight_address <= weight_address + 1;
                end

                // Only need to load the scale and bias during the first clockcycle of a new output block
                if (load_scale_and_bias == 1'b1) begin
                    load_scale_and_bias <= 1'b0;
                end

                if (~processing_residual_part) begin
                    if (~is_odd_layer_number) begin
                        prev_input_address <= input_address_per_layer[layer];
                    end

                    if (input_address_per_layer[layer] == blocks_times_kernel_for_layer & ~(is_true_input_layer & load_inputs_from_activation_memory)) begin
                        input_address_per_layer[layer] <= 0;
                    end else begin
                        input_address_per_layer[layer] <= input_address_per_layer[layer] + 1;
                    end

                    // If all inputs from all previous layer blocks have been processed into the current layer, current block
                    if (processed_all_input_blocks_into_current_block) begin
                        steps_in_input_layer <= 0;
                        bias_address <= bias_address + 1;
                        load_scale_and_bias <= 1'b1;  // Only set it to 1 when the output address changes

                        // Also set it to zero when we have completed all out blocks of a linear layer, as linear layers dont have rotating input buffers
                        if (out_address_per_layer[layer] == blocks_times_kernel_for_next_layer) begin
                            out_address_per_layer[layer] <= 0;
                        end else begin
                            out_address_per_layer[layer] <= out_address_per_layer[layer] + 1;
                        end

                        // If one complete kernel for the current layer is done
                        if (out_blocks_completed_counter == (is_output_layer_internal & use_few_shot_next_layer_blocks ? few_shot_next_layer_blocks : blocks_per_layer[layer+1])) begin
                            out_blocks_completed_counter <= 0;

                            // If we are processing a linear layer, the shift times blocks per layer is 0 so we can skip this
                            if (~processing_linear_layer) begin
                                input_address_per_layer[layer] <= 1 + input_address_per_layer[layer] + shift_times_blocks_per_layer - (input_address_per_layer[layer] + shift_times_blocks_per_layer >= blocks_times_kernel_for_layer ? blocks_times_kernel_for_layer + 1 : 0);
                            end

                            // If we continue into the next layer, there is no need to reset the weight address
                            // as it will need to increase to access the next layer's weights
                            if (steps_in_kernel_counter[layer] + 1 == kernel_size_per_layer[layer+1]) begin
                                steps_in_kernel_counter[layer] <= steps_in_kernel_counter[layer] + 1 - re_eval_count;

                                // If we have completed also the last linear layer, jump back to the first, TCN layer
                                if (layer + 1 == num_conv_and_linear_layers) begin
                                    layer <= 0;
                                    weight_address <= 0;
                                    bias_address <= 0;
                                end else begin
                                    // If we are in a linear layer of block size 1, the next linear layer will immediately access
                                    // the output from this layer, which will not be ready yet at that point, so we have to let
                                    // the system wait for that to be ready
                                    if (blocks_per_layer[layer+1] <= 1 && kernel_size_per_layer[layer+1] == 1) begin
                                        // If the next layer is residual and two residual steps are required before using the inputs
                                        // from the previous layer, then we do not have to wait for the output to be stored
                                        // NOTE: this line can optionally be removed as it only saves one clock cycle in a very specific case
                                        if (!(is_next_layer_residual & blocks_per_layer[layer] > 0 & force_downsample))
                                            dont_wait_for_linear_layer <= 1'b0;
                                    end

                                    if (is_next_layer_residual) begin
                                        processing_residual_part <= 1'b1;
                                    end

                                    layer <= layer + 1;
                                end
                            end else begin
                                steps_in_kernel_counter[layer] <= steps_in_kernel_counter[layer] + 1;

                                // This will only get triggered when we complete the last layer of the TCN
                                // Statement written in such a way that is it only triggered when a linear
                                // layer is present
                                if (layer + 1 == num_conv_layers && has_one_or_more_linear_layers_at_the_end) begin
                                    layer <= layer + 1;

                                    // TODO: there should be a nicer way to do this and the other dont_wait_for_linear_layer
                                    if (blocks_per_layer[layer+1] == 0 || blocks_per_layer[layer+1] == 1) dont_wait_for_linear_layer <= 1'b0;
                                end else begin
                                    layer <= 0;
                                    weight_address <= 0;
                                    bias_address <= 0;
                                end
                            end
                        end else begin
                            out_blocks_completed_counter <= out_blocks_completed_counter + 1;

                            if (is_odd_layer_number & ~processing_linear_layer) begin
                                processing_residual_part <= 1'b1;
                            end
                        end
                    end else begin
                        steps_in_input_layer <= steps_in_input_layer + 1;
                    end
                end else if (residual_steps == blocks_per_layer[layer-1]) begin
                    processing_residual_part <= 1'b0;
                    residual_steps <= 0;
                end else begin
                    residual_steps <= residual_steps + 1;

                    if (~force_downsample & current_layer_blocks_same_as_previous_layer_blocks) begin
                        processing_residual_part <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
