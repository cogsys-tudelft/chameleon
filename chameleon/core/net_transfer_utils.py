from pathlib import Path
from typing import List, Optional, OrderedDict, Union, Tuple
import math

import numpy as np

from asic_cells.utils import to_binary_string, chunk_list, flatten_list_of_lists

from brevitas_utils import load_quant_state_dict

from chameleon.core.shared_utils import flog2, clog2
from chameleon.core.net_load_utils import get_quant_layers


def twos_complement(x: int, num_bits: int):
    """Return the two's complement of x with num_bits bits.

    Args:
        x (int): The number to take the two's complement of.
        num_bits (int): The number of bits to use for the two's complement.

    Returns:
        int: The two's complement of x with num_bits bits.
    """

    if x > 2**(num_bits - 1) - 1:
        raise ValueError(f"x ({x}) is too large to represent with {num_bits} bits")
    
    if x < -2**(num_bits - 1):
        raise ValueError(f"x ({x}) is too small to represent with {num_bits} bits")

    return (x + (1 << num_bits)) % (1 << num_bits)


def pad_weight(weight: np.ndarray, padding_value: int, pe_array_size: int) -> np.ndarray:
    if len(weight.shape) == 2:
        weight = np.expand_dims(weight, 2)

    # Pad the weight tensor with the padding value to the closest multiple of the PE array size
    input_channels, output_channels, kernel_size = weight.shape

    new_input_channels = math.ceil(input_channels / pe_array_size) * pe_array_size
    new_output_channels = math.ceil(output_channels / pe_array_size) * pe_array_size

    padded_weight = np.full((new_input_channels, new_output_channels, kernel_size), padding_value)
    padded_weight[:input_channels, :output_channels, :] = weight

    return padded_weight


def get_weight_rows(weight: np.ndarray, pe_array_size: int = 16, subsection_size: int = 4, padding_value: Optional[int] = None, subsection_weight: bool = False) -> list[np.ndarray]:
    """Get flattened rows of weights in the correct format to be sent to the ASIC.

    Args:
        weight (np.ndarray): Non-flattened original 3D weight tensor of shape (input_channels, output_channels, kernel_size)
        pe_array_size (int, optional): PE array size. Defaults to 16.
        subsection_size (int, optional): PE array sub row size; set to -1 to not use a sub row memory layout. Defaults to 4.
        padding_value (Optional[int], optional): Value to pad the weight tensor with to be a multiple of the pe_array_size. Defaults to None.
        subsection_weight (bool, optional): Whether the weight is a subsection network. Defaults to False.

    Returns:
        list[np.ndarray]: List of flattened rows of weights
    """

    if len(weight.shape) == 2:
        weight = np.expand_dims(weight, 2)

    weight = np.transpose(weight, (1, 0, 2))

    true_pe_array_size = subsection_size if subsection_weight else pe_array_size

    if padding_value is not None:
        weight = pad_weight(weight, padding_value, true_pe_array_size)

    input_channels, output_channels, kernel_size = weight.shape

    assert input_channels % true_pe_array_size == 0, f"The number of input channels ({input_channels}) must be a multiple of {true_pe_array_size}, the PE array size or subsection size"
    assert output_channels % true_pe_array_size == 0, f"The number of output channels ({output_channels}) must be a multiple of {true_pe_array_size}, the PE array size or subsection size"
    
    if subsection_weight:
        # Calculate number of chunks along each dimension
        num_chunks_x = weight.shape[0] // subsection_size
        num_chunks_y = weight.shape[1] // subsection_size

        # Initialize an array to hold the padded chunks
        result_shape = (num_chunks_x * pe_array_size, num_chunks_y * pe_array_size, weight.shape[2])
        result = np.zeros(result_shape, dtype=weight.dtype)

        # Process each chunk, pad, and place in the result array
        for i in range(num_chunks_x):
            for j in range(num_chunks_y):
                chunk = weight[i*subsection_size:(i+1)*subsection_size, j*subsection_size:(j+1)*subsection_size, :]
                padded_chunk = np.pad(chunk, ((0, pe_array_size - chunk.shape[0]), (0, pe_array_size - chunk.shape[1]), (0, 0)), mode='constant')
                result[i*pe_array_size:(i+1)*pe_array_size, j*pe_array_size:(j+1)*pe_array_size, :] = padded_chunk

        weight = result

    input_channels, output_channels, kernel_size = weight.shape

    assert input_channels % true_pe_array_size == 0, "The number of input channels must be a multiple of the PE array size"
    assert output_channels % true_pe_array_size == 0, "The number of output channels must be a multiple of the PE array size"
    
    rows = []
    
    for output_block in range(output_channels // pe_array_size):
        for kernel in range(kernel_size):
            for input_block in range(input_channels // pe_array_size):
                current_block = weight[input_block*pe_array_size:(input_block+1)*pe_array_size, output_block*pe_array_size:(output_block+1)*pe_array_size, kernel]

                if subsection_size == -1:
                    row = current_block.T.flatten()
                else:
                    # Apply corrections for the weight layout as required by subsection operation mode
                    sub_block = current_block[:subsection_size, :]
                    remaining = current_block[subsection_size:, :]
                    
                    row = np.concatenate((sub_block.T.flatten(), remaining.T.flatten()))

                rows.append(row)

    return rows


def get_bias_rows(bias: np.ndarray, pe_array_size: int = 16, subsection_size: int = 4, padding_value: Optional[int] = None, subsection_bias: bool = False) -> list[np.ndarray]:
    """Get rows of biases in the correct format to be sent to the ASIC.

    Args:
        bias (np.ndarray): Non-flattened original bias tensor of shape (channels,)
        pe_array_size (int, optional): PE array size. Defaults to 16.
        padding_value (Optional[int], optional): Value to pad the bias tensor with to be a multiple of the pe_array_size. Defaults to None.

    Returns:
        list[np.ndarray]: List of rows of biases
    """

    padding_size = subsection_size if subsection_bias else pe_array_size

    if padding_value is not None:
        # Pad the bias tensor with the padding value to the closest multiple of the PE array size
        channels = bias.shape[0]

        new_channels = math.ceil(channels / padding_size) * padding_size

        padded_bias = np.full((new_channels,), padding_value)
        padded_bias[:channels] = bias

        bias = padded_bias

    channels, = bias.shape

    assert channels % padding_size == 0, "The number of channels must be a multiple of the PE array size or subsection size"

    if subsection_bias:
        chunk_size = subsection_size

        chunks = np.array_split(bias, np.arange(chunk_size, len(bias), chunk_size))
        zeroes = np.zeros(pe_array_size - chunk_size, dtype=bias.dtype)

        result = []

        # Iterate over chunks and append zeroes in between
        for chunk in chunks:
            result.extend(chunk)
            result.extend(zeroes)

        # Convert result back to a numpy array and add zeroes at the end
        bias = np.array(result, dtype=bias.dtype)
    
    channels, = bias.shape

    assert channels % padding_size == 0, "The number of channels must be a multiple of the PE array size"

    rows = []

    for output_block in range(channels // pe_array_size):
        rows.append(bias[output_block*pe_array_size:(output_block+1)*pe_array_size])

    return rows


def get_input_blocks(inputs: np.ndarray, channels_per_block: int = 4, padding_value: Optional[int] = None) -> list[np.ndarray]:
    """Splits the input array into blocks of size `channels_per_block` and returns a list of these blocks.

    Parameters:
        inputs (np.ndarray): The input array of shape (channels, time_steps)
        channels_per_block (int): The number of channels per input block (in other words: how many channels can you sent in a single transfer to the ASIC). Defaults to 4.
        padding_value (Optional[int], optional): Value to pad the input tensor with to be a multiple of the channels_per_block. Defaults to None.

    Returns:
        list[np.ndarray]: A list of numpy arrays, each representing a block of size `channels_per_block`
    """

    if padding_value is not None:
        # Pad the input tensor with the padding value to the closest multiple of the channels per block
        channels, time_steps = inputs.shape

        new_channels = math.ceil(channels / channels_per_block) * channels_per_block

        padded_inputs = np.full((new_channels, time_steps), padding_value)
        padded_inputs[:channels, :] = inputs

        inputs = padded_inputs

    channels, time_steps = inputs.shape

    assert channels % channels_per_block == 0, "The number of channels must be a multiple of the number of channels per block"

    blocks = []

    for t in range(time_steps):
        for output_block in range(channels // channels_per_block):
            blocks.append(inputs[output_block*channels_per_block:(output_block+1)*channels_per_block, t])

    return blocks


def bias_rows_to_messages(biases: List[np.ndarray],
                      pe_array_size: int = 16,
                      subsection_size: int = 4,
                      bias_bit_width: int = 15,
                      spi_message_bit_width: int = 32,
                      padding_value: Optional[int] = None,
                      subsection_bias: bool = False):
    all_bias_messages = []
    rows = []

    closest_pot_width = 2**int(math.ceil(math.log2(bias_bit_width * pe_array_size)))
    padding_width = closest_pot_width - bias_bit_width * pe_array_size

    for bias in biases:
        rows += get_bias_rows(bias, pe_array_size, subsection_size, padding_value, subsection_bias)

    for row in rows:
        messages = reversed(chunk_list("0" * padding_width + ''.join([to_binary_string(twos_complement(x, bias_bit_width), bias_bit_width) for x in reversed(row)]), spi_message_bit_width))

        all_bias_messages += messages

    all_bias_messages = [int(x, 2) for x in all_bias_messages]

    return all_bias_messages


def weight_rows_to_messages(rows: List[List], weight_bit_width: int = 4, spi_message_bit_width: int = 32):
    # weights order is: [temporal block1, optional residual block, temporal block 2]... + linear layers
    all_weight_messages = []

    for row in rows:
        messages = chunk_list(''.join(map(lambda x: to_binary_string(x, weight_bit_width), row)), spi_message_bit_width)

        messages_split = list(
            map(lambda x: int(x, 2), map(lambda x: ''.join(reversed(chunk_list(x, weight_bit_width))), messages))
        )

        all_weight_messages += messages_split

    return all_weight_messages


def activations_to_messages(activations: np.ndarray,
                            activation_bit_width: int = 4,
                            in_subsection_mode: bool = False,
                            pe_array_size: int = 16,
                            subsection_size: int = 4,
                            spi_message_bit_width: int = 32):
    rows = activations.T.reshape(-1, subsection_size if in_subsection_mode else pe_array_size)

    if in_subsection_mode:
        # Right pad with zeros until pe_array_size
        rows = np.pad(rows, ((0, 0), (0, pe_array_size - subsection_size)), mode='constant')

    all_activation_messages = []

    for row in rows:
        messages = chunk_list(''.join(map(lambda x: to_binary_string(x, activation_bit_width), row)), spi_message_bit_width)

        messages_split = list(
            map(lambda x: int(x, 2), map(lambda x: ''.join(reversed(chunk_list(x, activation_bit_width))), messages))
        )

        all_activation_messages += messages_split
  
    return all_activation_messages


def get_input_messages(input_tensor: np.ndarray, input_bit_width: int = 4, channels_per_block: int = 4, padding_value: Optional[int] = None):
    all_input_messages = []

    blocks = get_input_blocks(input_tensor, channels_per_block, padding_value)

    for block in blocks:
        all_input_messages.append(''.join(map(lambda x: to_binary_string(x, input_bit_width), reversed(block))))

    return [int(message, 2) for message in all_input_messages]


def get_output_data(output_messages: List[int], output_bit_width: int = 4, channels_per_block: int = 2):
    bin_data_out = [bin(entry)[2:] for entry in output_messages]
    padded_bin_data_out = [entry.zfill(output_bit_width*channels_per_block) for entry in bin_data_out]
    matrix_format_output = list(map(lambda x: list(reversed(chunk_list(x, output_bit_width))), padded_bin_data_out))
    flattened_output = flatten_list_of_lists(matrix_format_output)
    decimal_outputs = list(map(lambda x: int(x, 2), flattened_output))

    return np.array(decimal_outputs)


def get_network_config(conv_blocks: List[int],
                       conv_kernel_sizes: List[int],
                       linear_blocks: List[int],
                       input_blocks: int,
                       max_kernel_size: int,
                       max_blocks: int,
                       max_activations : int,
                       reserve_extra_linear_layer: bool = True,
                       icl_layers_shots: Optional[Tuple[int, int]] = None,
                       are_icl_shots_labeled: bool = False,
                       activation_memory_address: Optional[int] = None,
                       continued_learning: Optional[bool] = False):
    assert len(conv_kernel_sizes) == len(conv_blocks), "The number of convolutional layers must match the number of convolutional block sizes"
    assert len(conv_blocks) % 2 == 0, "The number of convolutional layers must be even"

    num_linear_layers = len(linear_blocks)
    num_conv_and_linear_layers = len(conv_kernel_sizes) + num_linear_layers

    # Need one entry for every conv layer and every linear layer (value 1)
    # TODO: think about the size of the linear layers! Is this correct right now?
    kernel_size_per_layer = conv_kernel_sizes + [1]*num_linear_layers + ([1] if reserve_extra_linear_layer else [])

    if max_kernel_size is not None:
        assert np.max(kernel_size_per_layer) <= max_kernel_size, "The maximum kernel size must be less than or equal to the specified maximum kernel size"

    # Need one entry for the input layer, every conv layer, and every linear layer
    blocks_per_layer = [input_blocks] + conv_blocks + linear_blocks

    if max_blocks is not None:
        assert np.max(blocks_per_layer) <= max_blocks, "The maximum number of blocks must be less than or equal to the specified maximum number of blocks"

    # Need one entry for the input layer, every conv layer, and every linear layer
    # Simply multiply entry i of blocks_per_layer with entry i of kernel_size_per_layer
    # Except the last i of blocks_per_layer
    blocks_per_layer_times_kernel_size = [x * y for x, y in zip(kernel_size_per_layer, blocks_per_layer)]
    blocks_per_layer_times_kernel_size_cumsum = [sum(blocks_per_layer_times_kernel_size[1:i]) for i in range(len(blocks_per_layer_times_kernel_size)+1)][1:]

    if len(blocks_per_layer_times_kernel_size_cumsum) > 0:
        max_cusmum_index = np.argmax(blocks_per_layer_times_kernel_size_cumsum)
        max_output_address = blocks_per_layer_times_kernel_size_cumsum[max_cusmum_index] + blocks_per_layer[max_cusmum_index + (not reserve_extra_linear_layer)] - 1

        assert max_output_address < max_activations, "One of the activations of the network will be written outside of the activation memory"

    blocks_per_layer = np.array(blocks_per_layer) - 1
    
    assert np.min(blocks_per_layer) >= 0, "The number of blocks per layer must be non-negative"
    blocks_per_layer = list(blocks_per_layer)

    config = {
        "num_conv_layers": len(conv_kernel_sizes),
        "kernel_size_per_layer": kernel_size_per_layer,
        "blocks_per_layer": blocks_per_layer,
        "input_blocks_times_kernel_size": blocks_per_layer_times_kernel_size[0]
    }

    if icl_layers_shots is not None:
        assert continued_learning == None, "Continued learning and ICL layers cannot be used at the same time"

        num_extra_icl_layers, icl_shots = icl_layers_shots

        memory_padding = (2 if are_icl_shots_labeled else 1) * icl_shots * (blocks_per_layer[-num_extra_icl_layers-1] + 1)

        blocks_per_layer_times_kernel_size_cumsum = np.array(blocks_per_layer_times_kernel_size_cumsum)
        blocks_per_layer_times_kernel_size_cumsum[num_conv_and_linear_layers-num_extra_icl_layers:] += memory_padding
        blocks_per_layer_times_kernel_size_cumsum = list(blocks_per_layer_times_kernel_size_cumsum)

        config["input_blocks_times_kernel_size_icl_head"] = (memory_padding + 1)
        config["num_conv_and_linear_layers_full_icl_net"] = num_conv_and_linear_layers
    else:
        num_extra_icl_layers = 0

        if type(continued_learning) == bool:
            config["num_conv_and_linear_layers_full_icl_net"] = num_conv_and_linear_layers - 1 if continued_learning else num_conv_and_linear_layers + 1

    config["num_conv_and_linear_layers"] = num_conv_and_linear_layers  - num_extra_icl_layers

    first_layer_input_start_address = activation_memory_address if activation_memory_address is not None else 0
    config["blocks_per_layer_times_kernel_size_cumsum"] = [first_layer_input_start_address] + blocks_per_layer_times_kernel_size_cumsum

    return config


def get_few_shot_scales(shots: int, weight_bit_width: int, activation_bit_width: int, max_shots: int, few_shot_scale: Optional[int] = None, k_shot_division_scale: Optional[int] = None):
    if shots > 0:
        max_weight_value = 2**(2**(weight_bit_width-1)-1)
        max_activation_value = 2**activation_bit_width-1

        if few_shot_scale == None:
            few_shot_scale = flog2(max_weight_value/max_activation_value/shots)

        left_shift_bit_width = clog2(max_activation_value*max_shots/max_weight_value)
        right_shift_bit_width = clog2(clog2(max_weight_value+1))
        max_bit_width = max(left_shift_bit_width, right_shift_bit_width) + 1

        if few_shot_scale < 0:
            few_shot_scale_cfg_value = abs(few_shot_scale)

            assert few_shot_scale_cfg_value < 2**right_shift_bit_width, "Few-shot scale is too large"

            few_shot_scale_cfg_value = (1 << (max_bit_width - 1)) + few_shot_scale_cfg_value
        else:
            assert few_shot_scale < 2**left_shift_bit_width, "Few-shot scale is too large"

            few_shot_scale_cfg_value = few_shot_scale

        if k_shot_division_scale == None:
            # TODO: should this be blog2 or clog2?
            k_shot_division_scale = flog2(shots)
    else:
        few_shot_scale_cfg_value = 0
        k_shot_division_scale = 0

    return few_shot_scale, (few_shot_scale_cfg_value, k_shot_division_scale)


def get_quant_state_dict_and_layers(path_or_state_dict: Union[str, Path, OrderedDict],
                                    slog2_weights: bool = True, scale_bit_width: int = 4,
                                    accepted_layers: Optional[List[str]] = None,
                                    n_last_layers_to_remove: Optional[int] = None):

    if isinstance(path_or_state_dict, (str, Path)):
        quant_state_dict = load_quant_state_dict(path_or_state_dict)
    else:
        quant_state_dict = path_or_state_dict

    quant_layers = get_quant_layers(quant_state_dict, slog2_weights, scale_bit_width, accepted_layers)

    if n_last_layers_to_remove != None and n_last_layers_to_remove != 0:
        quant_layers = quant_layers[:-n_last_layers_to_remove]

    return quant_state_dict['in_quant.act_quant'], quant_layers
