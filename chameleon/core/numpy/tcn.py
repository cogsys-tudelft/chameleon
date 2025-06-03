from typing import Union, List

import numpy as np

from chameleon.core.quant_conversions import slog2_to_int
from chameleon.core.shared_utils import QuantLayers


def get_receptive_field_size(kernel_size: Union[int, List[int]],
                             num_layers: int,
                             dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN. We assume the TCN structure of the paper
    from Bai et al.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (Union[int, List[int]]): Size of the kernel(s).
        num_layers (int): Number of layers in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_layers

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size[l-1] - 1)
        for l in range(1,num_layers+1)
    ]) + 1


def relu_and_scale(input_tensor: np.ndarray, scale: int, max_activation_value: int, max_accumulation_value: int):
    if max_accumulation_value != -1:
        assert input_tensor.max() < max_accumulation_value
        assert input_tensor.min() >= -max_accumulation_value

    # If scale is -1, we do not have output quantization
    if scale == -1:
        return input_tensor

    out = np.maximum(input_tensor, 0)
    out = np.right_shift(out, scale)

    if max_activation_value != -1:
        # Truncate all values greater than the maximum activation value
        out = out & (max_activation_value-1)

    return out


def temporal_conv1d(input_tensor: np.ndarray, weight_tensor: np.ndarray, dilation: int = 1, bias_tensor = None):
    output_channels, input_channels, kernel_size = weight_tensor.shape
    data_input_channels, length = input_tensor.shape

    assert data_input_channels == input_channels
    assert length >= kernel_size

    effective_kernel_size = (kernel_size-1)*(dilation-1)+kernel_size
    output_length = length-effective_kernel_size
    output_length = (output_length // dilation if output_length >= 1 else 0) + 1

    output_tensor = np.zeros((output_length, output_channels), dtype=input_tensor.dtype)

    for i in range(effective_kernel_size-1, length, dilation):
        start_idx = i-(effective_kernel_size-1)
        output_tensor[start_idx//dilation] = np.tensordot(weight_tensor, input_tensor[:, start_idx:start_idx+kernel_size*dilation:dilation], axes=2)

    output_tensor = output_tensor.T
    
    # Add (dilation-1) number of zeros between each output timestep
                    
    if dilation == 1:
        return output_tensor
                    
    dilated_output_tensor = np.zeros((output_channels, output_length+(dilation-1)*(output_length-1)), dtype=input_tensor.dtype)

    count = 0

    for i in range(dilated_output_tensor.shape[1]):
        if i % dilation == 0:
            dilated_output_tensor[:, i] = output_tensor[:, count]
            count += 1
        else:
            if bias_tensor is not None:
                dilated_output_tensor[:, i] = -bias_tensor
            else:
                dilated_output_tensor[:, i] = 0

    return dilated_output_tensor


def temporal_conv1d_bias_relu(input_tensor: np.ndarray, weight_tensor: np.ndarray, bias_tensor: np.ndarray, scale: int, max_activation_value: int, max_accumulation_value: int, dilation: int = 1):
    assert bias_tensor.shape == (weight_tensor.shape[0],)

    output_tensor = temporal_conv1d(input_tensor, weight_tensor, dilation, bias_tensor) + np.expand_dims(bias_tensor, 1)

    return relu_and_scale(output_tensor, scale, max_activation_value, max_accumulation_value)


def conv_1x1(input_tensor: np.ndarray, weight_tensor: np.ndarray) -> np.ndarray:
    _, input_channels, kernel_size = weight_tensor.shape
    data_input_channels, _ = input_tensor.shape

    assert data_input_channels == input_channels
    assert kernel_size == 1

    return weight_tensor[:, :, 0] @ input_tensor


def fc(input_tensor: np.ndarray, weight_tensor: np.ndarray):
    assert input_tensor.shape[1] == 1, "Only a single timestep can be fed into a linear layer"

    return conv_1x1(input_tensor, np.expand_dims(weight_tensor, 2))


def fc_bias_relu(input_tensor: np.ndarray, weight_tensor: np.ndarray, bias: np.ndarray, scale: int, max_activation_value: int, max_accumulation_value: int):
    input_tensor = fc(input_tensor, weight_tensor) + np.expand_dims(bias, 1)

    return relu_and_scale(input_tensor, scale, max_activation_value, max_accumulation_value), input_tensor


def tcn_layer(input_tensor: np.ndarray, weight_tensor1: np.ndarray, weight_tensor2: np.ndarray, downsample_tensor: Union[np.ndarray, None], bias1: np.ndarray, bias2: np.ndarray, scale1: int, scale2: int, scale_res: int = 0, dilation: int = 1, max_activation_value: int = 16, max_accumulation_value: int = 20):
    assert bias2.shape == (weight_tensor2.shape[0],)

    intermediate_tensor = temporal_conv1d_bias_relu(input_tensor, weight_tensor1, bias1, scale1, max_activation_value, max_accumulation_value, dilation)
    output_tensor = temporal_conv1d(intermediate_tensor, weight_tensor2, dilation) + np.expand_dims(bias2, 1)
    residual_tensor = conv_1x1(input_tensor, downsample_tensor) if downsample_tensor is not None else input_tensor
    output_tensor += (np.right_shift if scale_res < 0 else np.left_shift)(residual_tensor[:, -output_tensor.shape[1]:], abs(scale_res))

    return relu_and_scale(output_tensor, scale2, max_activation_value, max_accumulation_value), intermediate_tensor, output_tensor


def tcn_network(input_tensor: np.ndarray,
                layers: QuantLayers,
                weight_bit_width: int = 4,
                act_bit_width: int = 4,
                accum_bit_width: int = 20,
                slog2_weights: bool = True):
    intermediate_tensors = []
    unscaled_output_tensor = None

    # bit_width -1 as the bias is signed
    max_accumulation_value = -1 if accum_bit_width == -1 else 2**(accum_bit_width-1)
    max_activation_value = -1 if act_bit_width == -1 else 2**act_bit_width

    # Support for 1D input tensors (in case of pure MLPs)
    if len(input_tensor.shape) == 1:
        input_tensor = np.expand_dims(input_tensor, 1)

    for i, layer in enumerate(layers):
        # If we encounter a linear layer
        if len(layer) == 3:
            weight, bias, scale = layer
            weight = slog2_to_int(weight, weight_bit_width) if slog2_weights else weight
            input_tensor, unscaled_output_tensor = fc_bias_relu(input_tensor,
                                                                weight, bias,
                                                                scale, max_activation_value,
                                                                max_accumulation_value)
            intermediate_tensors.append(input_tensor)
        elif len(layer) == 2:
            ((weight1, bias1, scale1), _), ((weight2, bias2, scale2), (downsample_weight, downsample_scale)) = layer
            weight1, = [(slog2_to_int(w, weight_bit_width) if w is not None else w) if slog2_weights else w for w in (weight1,)]
            weight2, = [(slog2_to_int(w, weight_bit_width) if w is not None else w) if slog2_weights else w for w in (weight2,)]
            downsample_weight, = [(slog2_to_int(w, weight_bit_width) if w is not None else w) if slog2_weights else w for w in (downsample_weight,)]
            input_tensor, intermediate_tensor, unscaled_output_tensor = tcn_layer(input_tensor,
                                                                                  weight1, weight2,
                                                                                  downsample_weight,
                                                                                  bias1, bias2,
                                                                                  scale1, scale2,
                                                                                  downsample_scale,
                                                                                  2**i, max_activation_value,
                                                                                  max_accumulation_value)
            intermediate_tensors.extend([intermediate_tensor, input_tensor])
        else:
            raise ValueError("Invalid layer configuration")

    # Remove last intermediate tensor (as it is also stored in the input_tensor)
    return input_tensor, intermediate_tensors[:-1], unscaled_output_tensor
