import numpy as np

from chameleon.core.quant_conversions import slog2_to_int


def left_right_shift(x: np.ndarray, shift: int, pos_is_left: bool = True):
    shift = shift if pos_is_left else -shift

    if shift >= 0:
        return np.left_shift(x, shift)
    else:
        return np.right_shift(x, -shift)


def compute_expected_weight_and_bias(ways: list[np.ndarray], weight_bit_width: int, few_shot_scale: int, use_l2_for_few_shot: bool, accumulation_bit_width: int = -1):
    ways_arr = slog2_to_int(np.array(ways), weight_bit_width)

    if use_l2_for_few_shot:
        bias = np.sum(ways_arr**2, axis=1)

        if accumulation_bit_width != -1:
            assert np.max(bias) <= 2**(2**(accumulation_bit_width-1)-1), "Bias is too large for the given bit width."

        bias = left_right_shift(bias, few_shot_scale)

        bias = -bias // 2
    else:
        bias = np.zeros(len(ways), dtype=int)

    return ways_arr, bias


def get_subsection_blocks_2d(array: np.ndarray, block_size: int, stride: int):
    # Initialize a new array to reconstruct the original array
    reconstructed_array = np.zeros((block_size*(array.shape[0] // stride), block_size*(array.shape[1] // stride)), dtype=array.dtype)

    # Iterate over the array to extract blocks
    for i in range(0, array.shape[0] - block_size + 1, stride):
        for j in range(0, array.shape[1] - block_size + 1, stride):
            block = array[i:i + block_size, j:j + block_size]

            isz = block_size*(i//stride)
            jsz = block_size*(j//stride)

            reconstructed_array[isz:isz + block_size, jsz:jsz + block_size] = block

    return reconstructed_array


def get_subsection_blocks_1d(array: np.ndarray, block_size: int, stride: int):
    # Initialize a new array to reconstruct the original array
    reconstructed_array = np.zeros((block_size*(array.shape[0] // stride),), dtype=array.dtype)

    # Iterate over the array to extract blocks
    for i in range(0, array.shape[0] - block_size + 1, stride):
            block = array[i:i + block_size]
            isz = block_size*(i//stride)
            reconstructed_array[isz:isz + block_size] = block

    return reconstructed_array 
