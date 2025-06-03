import cocotb

from .utils import verify_net_structure


RANDOM_SEED = 2
CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
SAME_STRUCTURE_REPEATS = 2
NUM_FORWARD_PASSES = 7
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"

CONV_KERNEL_SIZES = []
CONV_BLOCKS = []
SHOTS = 0
IS_NEW_TASK = False


async def run_fc_mlp_test(dut, input_blocks: int, linear_blocks: list[int]):
    # TODO: make sure all processing options are considered here
    await verify_net_structure(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        input_blocks, CONV_BLOCKS,
        CONV_KERNEL_SIZES,
        linear_blocks,
        in_subsection_mode=None, clock_freq=CLOCK_FREQ,
        num_structure_repeats=SAME_STRUCTURE_REPEATS,
        num_forward_passes=NUM_FORWARD_PASSES,
        is_new_task=IS_NEW_TASK, shots=SHOTS,
        slog2_weights=SLOG2_WEIGHTS, seed=RANDOM_SEED)


@cocotb.test()
async def single_block_in_single_block_out_single_linear_layer(dut):
    input_blocks = 1
    linear_blocks = [1]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def single_block_in_multiple_block_out_single_linear_layer(dut):
    input_blocks = 1
    linear_blocks = [2]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def single_block_mlp_one_input_block(dut):
    input_blocks = 1
    linear_blocks = [1, 1]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def single_block_mlp_2x1(dut):
    input_blocks = 1
    linear_blocks = [2, 1]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def three_output_blocks(dut):
    input_blocks = 1
    linear_blocks = [1, 3]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def single_block_mlp_multiple_input_blocks(dut):
    input_blocks = 3
    linear_blocks = [1, 1]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def multiple_block_in_single_block_out_single_linear_layer(dut):
    input_blocks = 2
    linear_blocks = [1]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def mlp_with_four_layers(dut):
    input_blocks = 2
    linear_blocks = [3, 4, 3, 2]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def mlp_with_1x1_in_the_middle(dut):
    input_blocks = 2
    linear_blocks = [2, 1, 1, 2]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)


@cocotb.test()
async def max_layer_size(dut):
    input_blocks = 1
    linear_blocks = [64]

    await run_fc_mlp_test(dut, input_blocks, linear_blocks)
