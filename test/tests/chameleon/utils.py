from typing import Optional, Tuple, Union
import inspect

from chameleon.sim.chameleon_sim_controller import ChameleonSimController
from chameleon.core.chameleon_interface import ChameleonParams


PASSED_TEST_FILE_NAME = "passed.txt"


def caller_name():
    frame = inspect.currentframe().f_back.f_back
    return frame.f_code.co_name


def get_params_and_controller(dut, cfg_memory_file: str, pointer_file: str):
    param_args = {}

    is_struct = getattr(dut, "CLK_EXT_PAD", None) is not None

    if not is_struct:
        for arg in ChameleonParams.__annotations__:
            param_args[arg] = int(getattr(dut, arg))

        params = ChameleonParams(**param_args)
    else:
        params = ChameleonParams(
            MESSAGE_BIT_WIDTH=32,
            CODE_BIT_WIDTH=4,
            START_ADDRESS_BIT_WIDTH=16,
            ACTIVATION_BIT_WIDTH=4,
            SCALE_BIT_WIDTH=4,
            BIAS_BIT_WIDTH=14,
            PE_COLS=16,
            PE_ROWS=16,
            SUBSECTION_SIZE=4,
            WEIGHT_BIT_WIDTH=4,
            HIGH_SPEED_IN_PINS=16,
            HIGH_SPEED_OUT_PINS=8,
            ACTIVATION_ROWS=256,
            WEIGHT_ROWS=1024,
            BIAS_ROWS=128,
            INPUT_ROWS=32,
            ACCUMULATION_BIT_WIDTH=18,
            MAX_KERNEL_SIZE=15,
            MAX_NUM_CHANNELS=1024,
            FEW_SHOT_ACCUMULATION_BIT_WIDTH=18,
            MAX_SHOTS=127,
            MAX_NUM_LOGITS=1024
        )

    controller = ChameleonSimController(dut, params, cfg_memory_file, pointer_file, is_struct)

    return params, controller


async def verify_net_structure(dut,
                               cfg_memory_file: str,
                               pointer_file: str,
                               input_blocks: int,
                               conv_blocks: list[int],
                               conv_kernel_sizes: list[int],
                               linear_blocks: list[int],
                               clock_freq: int,
                               num_structure_repeats: int = 1,
                               num_forward_passes: int = 1,
                               is_new_task: bool = False,
                               shots: int = 0,
                               slog2_weights: bool = True,
                               seed: int = 0,
                               in_subsection_mode: Optional[bool] = None,
                               force_downsample: Optional[bool] = None,
                               check_memory_contents: bool = False,
                               num_continuous_forward_passes: int = 2,
                               classification_options: Union[str, bool] = 'both',
                               activation_memory_address: Optional[int] = None):
    _, sim_controller = get_params_and_controller(dut, cfg_memory_file, pointer_file)
    await sim_controller.start_and_verify_asic(1, clock_freq, reset_host=False, verify_memories=False)

    await sim_controller.verify_net_structure(
        input_blocks,
        conv_blocks,
        conv_kernel_sizes,
        linear_blocks,
        num_structure_repeats,
        num_forward_passes,
        is_new_task,
        shots,
        slog2_weights,
        seed,
        in_subsection_mode,
        force_downsample,
        check_memory_contents,
        num_continuous_forward_passes,
        classification_options,
        activation_memory_address
    )

    with open(PASSED_TEST_FILE_NAME, "a") as f:
        f.write(f"{caller_name()}\n")


async def learn_with_few_shots(dut,
                                    cfg_memory_file: str,
                                    pointer_file: str,
                                    dataset,
                                    shots: int,
                                    query_shots: int,
                                    ways: int,
                                    num_batches: int,
                                    ways_for_continued_learning: int,
                                    clock_freq: int,
                                    expected_accuracies: Tuple[float, float],
                                    quant_state_dict_file_path: str,
                                    n_last_layers_to_remove: Optional[int] = None,
                                    require_single_chunk: bool = False,
                                    seed: int = 0,
                                    l2_options: Union[str, bool] = 'both',
                                    check_memory_contents: bool = True,
                                    in_subsection_mode: bool = False,
                                    icl_shots: int = 0,
                                    are_icl_shots_labeled: bool = False):
    _, c = get_params_and_controller(dut, cfg_memory_file, pointer_file)
    await c.start_and_verify_asic(1, clock_freq, reset_host=False, verify_memories=False)

    await c.learn_with_few_shots(
        dataset,
        shots,
        query_shots,
        ways,
        num_batches,
        ways_for_continued_learning,
        expected_accuracies,
        quant_state_dict_file_path,
        n_last_layers_to_remove,
        require_single_chunk,
        seed,
        l2_options,
        check_memory_contents,
        in_subsection_mode,
        icl_shots,
        are_icl_shots_labeled
    )

    with open(PASSED_TEST_FILE_NAME, "a") as f:
        f.write(f"{caller_name()}\n")
