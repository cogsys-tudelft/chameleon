`include "states.vh"

module few_shot_learning_control #(
    parameter integer SHOT_BIT_WIDTH = 5,
    parameter integer STATE_BIT_WIDTH = 3,
    parameter integer ACTIVATION_ADDRESS_WIDTH = 8,
    parameter integer WEIGHT_ADDRESS_WIDTH = 8,
    parameter integer BIAS_ADDRESS_WIDTH = 4,
    parameter integer BLOCKS_WIDTH = 4,
    parameter integer PE_COLS = 16,
    parameter integer SUBSECTION_SIZE = 4,
    parameter integer LAYER_WIDTH = 4,
    parameter integer CUMSUM_WIDTH = 4,
    parameter integer WAYS_BIT_WIDTH = 8,
    localparam integer FewShotShiftBitWidth = $clog2(PE_COLS)
) (
    input clk,
    input rst,

    input [STATE_BIT_WIDTH-1:0] state,

    input [ LAYER_WIDTH-1:0] num_conv_and_linear_layers,
    input [ LAYER_WIDTH-1:0] num_conv_and_linear_layers_full_icl_net,
    input [CUMSUM_WIDTH-1:0] cumsum_conv_layers,
    input [BLOCKS_WIDTH-1:0] linear_layer_blocks,

    input is_new_task_sync,
    input classification,
    input continued_learning,
    input in_context_learning,
    input skip_sending,
    input toggle_processing_new,
    input is_output_layer,
    input use_subsection,

    input [BLOCKS_WIDTH-1:0] embedding_layer_blocks,

    input [SHOT_BIT_WIDTH-1:0] shots,

    input [WEIGHT_ADDRESS_WIDTH-1:0] max_weight_address,
    input [  BIAS_ADDRESS_WIDTH-1:0] max_bias_address,

    input  [ACTIVATION_ADDRESS_WIDTH-1:0] in_activation_address,
    output [ACTIVATION_ADDRESS_WIDTH-1:0] in_activation_address_few_shot_corrected,

    input  [ACTIVATION_ADDRESS_WIDTH-1:0] out_activation_address,
    output [ACTIVATION_ADDRESS_WIDTH-1:0] out_activation_address_few_shot_corrected,

    output [WEIGHT_ADDRESS_WIDTH-1:0] weight_address_few_shot_corrected,
    output [  BIAS_ADDRESS_WIDTH-1:0] bias_address_few_shot_corrected,

    output reg processing_new_task_samples,
    output reg enable_activation_memory_for_few_shot,
    output enable_pe_array,
    output start_few_shot_processing,
    output reg done_few_shot_processing,
    output reg scale_and_bias_ready_few_shot,
    output reg enable_weight_memory_for_few_shot,
    output enable_bias_memory_for_few_shot,
    output configured_for_few_shot_processing,
    output use_few_shot_next_layer_blocks,
    output reg [WAYS_BIT_WIDTH-1:0] ways_received,
    output reg [SHOT_BIT_WIDTH-1:0] shots_received,
    output [WAYS_BIT_WIDTH-FewShotShiftBitWidth-1:0] weight_bias_offset,

    output [FewShotShiftBitWidth-1:0] few_shot_shift_required,

    output [LAYER_WIDTH-1:0] num_conv_and_linear_layers_few_shot_corrected,
    output reg [BLOCKS_WIDTH-1:0] blocks_per_layer_few_shot_corrected
);

    // Local parameters ---------------------------------------------------------------------------

    localparam integer FewShotShiftBitWidthSubsection = $clog2(SUBSECTION_SIZE);

    // Registers ----------------------------------------------------------------------------------

    reg [ACTIVATION_ADDRESS_WIDTH-1:0] which_output_block;
    reg [SHOT_BIT_WIDTH-1:0] which_shot;
    reg [WEIGHT_ADDRESS_WIDTH-1:0] weight_address_offset;

    // Wires -------------------------------------------------------------------------------------

    wire received_all_shots;
    wire completed_all_output_blocks;
    wire enable_bias_memory_for_few_shot_early;
    wire correct_out_address;
    wire shots_is_non_zero;
    wire configured_for_prototype_learning;
    wire [BLOCKS_WIDTH:0] embedding_layer_blocks_plus_one;

    // Assignments --------------------------------------------------------------------------------

    assign received_all_shots = shots_received + 1 == shots;
    assign completed_all_output_blocks = which_output_block == embedding_layer_blocks;
    assign enable_bias_memory_for_few_shot_early = enable_weight_memory_for_few_shot && completed_all_output_blocks;
    assign embedding_layer_blocks_plus_one = embedding_layer_blocks + 1;
    assign shots_is_non_zero = shots != 0;
    assign configured_for_prototype_learning = classification && shots_is_non_zero;
    assign configured_for_few_shot_processing = configured_for_prototype_learning | in_context_learning;
    assign start_few_shot_processing = received_all_shots && skip_sending && ~in_context_learning;

    assign correct_out_address = is_output_layer && processing_new_task_samples;
    // ------
    // ------
    // ------
    // ------
    // TODO: COMPUTE BELOW VARIABLE IN AN EASIER WAY: out_activation_address
    // ------
    // ------
    // ------
    // ------
    // ------
    assign out_activation_address_few_shot_corrected = correct_out_address ? out_activation_address + shots_received * embedding_layer_blocks_plus_one : out_activation_address;
    assign in_activation_address_few_shot_corrected = (state == `PROCESSING_FEW_SHOT) ? cumsum_conv_layers + which_output_block + which_shot * embedding_layer_blocks_plus_one : in_activation_address;

    // Convert these wires to registers
    assign weight_bias_offset = use_subsection ? ways_received[WAYS_BIT_WIDTH-1:FewShotShiftBitWidthSubsection] : ways_received[WAYS_BIT_WIDTH-1:FewShotShiftBitWidth];
    assign few_shot_shift_required = use_subsection ? ways_received[FewShotShiftBitWidthSubsection-1:0] : ways_received[FewShotShiftBitWidth-1:0];
    assign weight_address_few_shot_corrected = which_output_block + weight_address_offset;
    assign bias_address_few_shot_corrected = max_bias_address + weight_bias_offset;

    assign use_few_shot_next_layer_blocks = configured_for_prototype_learning && ~processing_new_task_samples && ~in_context_learning;

    assign num_conv_and_linear_layers_few_shot_corrected = (shots_is_non_zero && ~processing_new_task_samples && in_context_learning) ? num_conv_and_linear_layers_full_icl_net : (configured_for_few_shot_processing ? (continued_learning ? (processing_new_task_samples ? num_conv_and_linear_layers_full_icl_net : num_conv_and_linear_layers) : (processing_new_task_samples ? num_conv_and_linear_layers : num_conv_and_linear_layers_full_icl_net)) : num_conv_and_linear_layers);

    // Child modules ------------------------------------------------------------------------------

    delay_register delay_register_enable_pe_array (
        .clk(clk),
        .rst(rst),
        .in(enable_activation_memory_for_few_shot),
        .in_delayed(enable_pe_array)
    );

    delay_register delay_register_scale_and_bias_ready_few_shot (
        .clk(clk),
        .rst(rst),
        .in(enable_activation_memory_for_few_shot && which_shot == 0),
        .in_delayed(scale_and_bias_ready_few_shot)
    );

    delay_register delay_register_enable_bias_memory_for_few_shot (
        .clk(clk),
        .rst(rst),
        .in(enable_bias_memory_for_few_shot_early),
        .in_delayed(enable_bias_memory_for_few_shot)
    );

    // Clocked logic ------------------------------------------------------------------------------

    always @(posedge clk) begin
        if (rst) begin
            processing_new_task_samples <= 1'b0;
            enable_activation_memory_for_few_shot <= 1'b0;
            ways_received <= 0;
            done_few_shot_processing <= 1'b0;
            enable_weight_memory_for_few_shot <= 1'b0;
            blocks_per_layer_few_shot_corrected <= 0;
            weight_address_offset <= 0;
            // Needs to be set to zero here as otherwise received_all_shots will become 'x'
            // and then will start_few_shot_processing will become 'x' as soon as skip_sending
            // goes high. Then, the FSM will be corrupted as that point states needs to change.
            shots_received <= 0;
            which_shot <= 0;
        end else if (done_few_shot_processing == 1'b1) begin
            done_few_shot_processing <= 1'b0;

            // "((ways_received+1)>>FewShotShiftBitWidth)" is the same as weight_bias_offset
            blocks_per_layer_few_shot_corrected <= (continued_learning ? linear_layer_blocks + 1 : 0) + (ways_received >> (use_subsection ? FewShotShiftBitWidthSubsection : FewShotShiftBitWidth));

            ways_received <= ways_received + 1;
        end else if (configured_for_few_shot_processing) begin
            if (state == `IDLE && toggle_processing_new && is_new_task_sync) begin
                processing_new_task_samples <= 1'b1;

                // Re-try functionality for re-inputting a sample for the current task
                if (shots_received != 0) begin
                    shots_received <= shots_received - 1;
                end else if (in_context_learning && shots_received == 0) begin
                    shots_received <= 1;
                end

                // Needs to be set to a non-zero value before we start the few-shot processing
                weight_address_offset <= max_weight_address + weight_bias_offset * embedding_layer_blocks_plus_one;
            end if (skip_sending) begin
                if (received_all_shots) begin
                    if (~in_context_learning) begin
                        enable_activation_memory_for_few_shot <= 1'b1;
                    end else begin
                        processing_new_task_samples <= 1'b0;
                    end

                    shots_received <= 0;

                    which_output_block <= 0;
                    which_shot <= 0;
                end else begin
                    shots_received <= shots_received + 1;
                end
            end else if (enable_weight_memory_for_few_shot == 1'b1) begin
                enable_weight_memory_for_few_shot <= 1'b0;

                if (completed_all_output_blocks) begin
                    which_output_block <= 0;
                    done_few_shot_processing <= 1'b1;
                    processing_new_task_samples <= 1'b0;
                end else begin
                    enable_activation_memory_for_few_shot <= 1'b1;
                    which_output_block <= which_output_block + 1;
                end
            end else if (state == `PROCESSING_FEW_SHOT) begin
                if (enable_activation_memory_for_few_shot == 1'b1) begin
                    if (which_shot + 1 == shots) begin
                        enable_activation_memory_for_few_shot <= 1'b0;
                    end else begin
                        which_shot <= which_shot + 1;
                    end
                end else begin
                    which_shot <= 0;
                    enable_weight_memory_for_few_shot <= 1'b1;
                end
            // Reset ways received by setting is_new_task high during idle (without setting toggle_processing_new high) after completing a way (meaning: having received all shots for that way)
            end else if (state == `IDLE && is_new_task_sync && ~toggle_processing_new && ~processing_new_task_samples) begin
                ways_received <= 0;
                blocks_per_layer_few_shot_corrected <= 0;
            end
        // If we started with few-shot learning (meaning, we received one or more shots) but we want to stop it (by setting number of shots to zero), then disable few shot processing
        end else if (state == `IDLE && processing_new_task_samples) begin
            processing_new_task_samples <= 1'b0;
            shots_received <= 0;
        end
    end
endmodule
