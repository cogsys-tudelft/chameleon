import cocotb

from .utils import verify_net_structure


RANDOM_SEED = 2
CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
SAME_STRUCTURE_REPEATS = 2
NUM_FORWARD_PASSES = 7
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"
# TODO: make sure all processing options are considered here
SHOTS = 0
IS_NEW_TASK = False


@cocotb.test()
async def single_tcn_layer_1_1x1_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_1_1x2_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [1, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_2_1x2_kernel_size_1(dut):
    input_blocks = 2
    kernel_size = 1
    conv_blocks = [1, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_2_1x2_kernel_size_1_force_downsample(dut):
    input_blocks = 2
    kernel_size = 1
    conv_blocks = [1, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        force_downsample=True,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_1_2x1x1x1_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [2,1,1,1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_1_2x1x2x1_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [2,1,2,1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def single_tcn_layer_1_2x1x2x2_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [2,1,2,2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)
    

@cocotb.test()
async def double_tcn_layer_all_one_blocks_kernel_size_1(dut):
    input_blocks = 1
    kernel_size = 1
    conv_blocks = [2, 3, 2, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        num_continuous_forward_passes=0)


@cocotb.test()
async def two_smallest_tcn_layers(dut):
    input_blocks = 1
    kernel_size = 3
    conv_blocks = [1, 1, 1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_all_one_blocks(dut):
    input_blocks = 1
    kernel_size = 3
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_all_one_blocks_k5(dut):
    input_blocks = 1
    kernel_size = 5
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)
    

@cocotb.test()
async def single_tcn_layer_all_one_blocks_k7(dut):
    input_blocks = 1
    kernel_size = 7
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_all_one_blocks_k9(dut):
    input_blocks = 1
    kernel_size = 9
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_all_one_blocks_k15(dut):
    input_blocks = 1
    kernel_size = 15
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_one_blocks_two_input_blocks(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [1, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_one_blocks_two_middle_blocks(dut):
    input_blocks = 1
    kernel_size = 3
    conv_blocks = [2, 1]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_one_blocks_two_output_blocks(dut):
    input_blocks = 1
    kernel_size = 3
    conv_blocks = [1, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True)
    

@cocotb.test()
async def single_tcn_layer_with_one_input_block(dut):
    input_blocks = 1
    kernel_size = 3
    conv_blocks = [2, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True)
    

@cocotb.test()
async def single_tcn_layer_two_blocks_one_middle_blocks(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [1, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_tcn_layer_one_block_middle_downsample(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [1, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True)


@cocotb.test()
async def single_tcn_layer_with_fc(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = [1]

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        num_continuous_forward_passes=0)


@cocotb.test()
async def single_tcn_layer_with_two_layer_mlp(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = [2, 1]

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        num_continuous_forward_passes=0)


@cocotb.test()
async def pure_residual_mlp(dut):
    input_blocks = 2
    kernel_size = 1
    conv_blocks = [2, 3, 2, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        num_continuous_forward_passes=0)



@cocotb.test()
async def double_tcn_layer_with_no_identity_possible(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3, 2, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        check_memory_contents=False)


@cocotb.test()
async def double_tcn_layer_with_two_identity_steps_possible(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 2, 2, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        check_memory_contents=False)


@cocotb.test()
async def triple_tcn_layer_with_mlp(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3, 2, 2, 3, 3]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = [2, 1]

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        check_memory_contents=False,
        num_continuous_forward_passes=0)


@cocotb.test()
async def triple_tcn_layer_same_block_last_first_layer(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [2, 3, 3, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=True,
        check_memory_contents=False)


@cocotb.test()
async def test_two_different_kernel_sizes_in_one_net(dut):
    input_blocks = 1
    conv_blocks = [1, 1, 1, 1]
    conv_kernel_sizes = [3, 3, 5, 5]
    linear_blocks = []

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        activation_memory_address=83)


@cocotb.test()
async def sc12_tcn_structure(dut):
    input_blocks = 2
    kernel_size = 3
    conv_blocks = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    conv_kernel_sizes = [kernel_size] * len(conv_blocks)
    linear_blocks = [1]

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=False,
        num_continuous_forward_passes=0)


@cocotb.test()
async def smaller_sc12_tcn_structure(dut):
    input_blocks = 2
    conv_blocks = [1, 1, 1, 1, 2, 2, 2, 2]
    conv_kernel_sizes = [5, 5, 3, 3, 3, 3, 3, 3]
    linear_blocks = [1]

    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED,
        force_downsample=False,
        num_continuous_forward_passes=0,
        activation_memory_address=100)
