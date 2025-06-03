from collections import OrderedDict
from typing import List, Optional, Dict, Union
import random
import math

import numpy as np
import numpy.testing as npt

from asic_cells.utils import chunk_list

from chameleon.core.shared_utils import clog2, QuantLayers
from chameleon.core.numpy.tcn import fc, temporal_conv1d, tcn_network, get_receptive_field_size
from chameleon.core.quant_conversions import slog2_to_int, float_to_int, float_to_slog2, float_to_uint


def get_true_weight_bit_width(bit_width: int, slog2_weights: bool):
    if slog2_weights:
        # Note: bit_width = 2 ** (true_weight_bit_width - 1), rewriting it yields:
        return int(math.log2(bit_width)) + 1

    return bit_width


def get_quant_layers(quant_state_dict: OrderedDict,
                     slog2_weights: bool = True,
                     scale_bit_width: int = 4,
                     accepted_layers: Optional[List[str]] = None) -> QuantLayers:
    # Split the state dict into activations and weights and biases
    activations = {}
    weights_and_biases = {}

    for name, properties in quant_state_dict.items():
        if name.endswith('.act_quant'):
            activations[name] = properties
        else:
            weights_and_biases[name] = properties

    # Get the weights, biases and scales in integer format
    conversion_func = float_to_slog2 if slog2_weights else float_to_int

    regular_params = []
    downsample_params = []

    for name, properties in weights_and_biases.items():
        weight = properties['weight']
        weight_scale = weight['scale']
        weight_value = weight['value']
        
        assert weight['signed'], "Only signed weights are supported for now"
        assert weight['zero_point'] == 0.0, "Only a zero point of 0 from a weight is supported for now"

        bias = properties['bias']
        bias_scale = bias['scale']
        bias_value = bias['value']

        assert bias['zero_point'] == 0.0, "Only a zero point of 0 from a bias is supported for now"

        true_weight_bit_width = get_true_weight_bit_width(weight['bit_width'], slog2_weights)

        int_weight = conversion_func(weight_value, weight_scale, bit_width=true_weight_bit_width)
        int_bias = float_to_int(bias_value, bias_scale, bit_width=bias['bit_width'], signed=True)

        entry = [name, int_weight, int_bias, weight_scale, bias_scale, weight['bit_width'], bias['bit_width']]

        if '.temp_layer' in name or '.fc' in name or (accepted_layers is not None and name in accepted_layers):
            regular_params.append(entry)
            downsample_params.append(None)
        elif '.downsample' in name:
            assert downsample_params[-1] == None, "Downsample layer already exists"

            downsample_params[-1] = entry
        else:
            raise ValueError(f"Unknown layer type: {name}")
        
    zero_corrected_biases = []
    output_corrected_scales = []
    downsample_scale_ratios = []
    num_post_linear_layers = 0

    has_output_quant = len(regular_params) + 1 == len(activations)

    assert len(activations) - has_output_quant == len(regular_params) == len(downsample_params), "Number of activations is not the same as the number of weight and biases"

    act_keys = list(activations)

    for i, ((name, weight, bias, weight_scale, bias_scale, weight_bit_width, bias_bit_width), ds) in enumerate(zip(regular_params, downsample_params)):
        is_fc = len(weight.shape) == 2
        nn_operation = fc if is_fc else temporal_conv1d
        num_post_linear_layers += is_fc

        kernel_size = 1 if is_fc else weight.shape[-1]
        input_channels = weight.shape[1]
        input_shape = (input_channels, kernel_size)

        true_weight_bit_width = get_true_weight_bit_width(weight_bit_width, slog2_weights)

        true_weight = slog2_to_int(weight, bit_width=true_weight_bit_width) if slog2_weights else weight
        zero_point_correction = nn_operation(np.full(input_shape, -activations[act_keys[i]]['zero_point']), true_weight).T[0]
        zero_corrected_biases.append(bias + zero_point_correction.astype(bias.dtype))

        if ds is not None:
            # If we need to incorporate a downsample layer
            _, ds_weight, ds_bias, _, ds_bias_scale, ds_weight_bit_width, _ = ds

            ds_scale_ratio = ds_bias_scale / bias_scale
            downsample_scale_ratios.append(clog2(ds_scale_ratio))

            input_channels_prev = ds_weight.shape[1]
            ds_input_shape = (input_channels_prev, 1)

            true_ds_weight_bit_width = get_true_weight_bit_width(ds_weight_bit_width, slog2_weights)

            ds_true_weight = slog2_to_int(ds_weight, bit_width=true_ds_weight_bit_width) if slog2_weights else ds_weight
            ds_zero_point_correction = nn_operation(np.full(ds_input_shape, -activations[act_keys[i-1]]['zero_point']), ds_true_weight).T[0]
            zero_corrected_biases[-1] += np.round(ds_scale_ratio * (ds_bias + ds_zero_point_correction)).astype(bias.dtype)
        elif i % 2 == 1 and not is_fc:
            # If we are dealing with a residual path without a downsample layer
            downsample_scale_ratios.append(clog2(activations[act_keys[i-1]]['scale']/bias_scale))
        else:
            # If we are not dealing with a downsample layer
            downsample_scale_ratios.append(None)

        scale_offset = 0

        if downsample_scale_ratios[-1] is not None:
            max_weight_value = 2**(2**(true_weight_bit_width-1)-1)

            # Perform correction for negative downsample scales where possible
            # by adjusting the quantized weights and biases of the second convolutional
            # block that combines with the residual path
            while downsample_scale_ratios[-1] < 0 and true_weight.max() < max_weight_value and true_weight.min() > -max_weight_value:
                zero_corrected_biases[-1] *= 2
                regular_params[i][1] += 1
                scale_offset = 1
                true_weight *= 2

                downsample_scale_ratios[-1] += 1

            assert downsample_scale_ratios[-1] >= 0, f"Downsample scale ratio must be greater than or equal to 0 for layer {i}. If this is the case for the first layer, this is likely due to very large input values in the float domain. Try to retrain your network with smaller input values."
            assert downsample_scale_ratios[-1] < 2**scale_bit_width, "Downsample scale ratio must be less than 2^(scale_bit_width-1)"

        assert zero_corrected_biases[-1].max() < 2**(bias_bit_width - 1), "Zero-corrected bias value is too large for the given bit width"
        assert zero_corrected_biases[-1].min() >= -2**(bias_bit_width - 1), "Zero-corrected bias value is too negative for the given bit width"

        if i == len(act_keys) - 1:
            corrected_scale = -1
        else:
            bias_corrected_scale = bias_scale / activations[act_keys[i+1]]['scale']
            clog2_bias_corrected_scale = clog2(bias_corrected_scale)

            npt.assert_almost_equal(2**clog2_bias_corrected_scale, bias_corrected_scale, decimal=7, verbose=True, err_msg=f"Corrected scale is not a power of 2 for layer {i}; this is likely the result of one weight or activation scale not being a power of 2")

            corrected_scale = -clog2_bias_corrected_scale + scale_offset

            assert corrected_scale >= 0, "Corrected scale must be greater than or equal to 0"
            assert corrected_scale < 2**scale_bit_width, "Corrected scale must be less than 2^scale_bit_width"

        output_corrected_scales.append(corrected_scale)

    regular_params_corrected = list(zip([p[1] for p in regular_params], zero_corrected_biases, output_corrected_scales))
    downsample_params_corrected = list(zip([p[1] if p is not None else p for p in downsample_params], downsample_scale_ratios))

    conv_params_only = regular_params_corrected
    downsample_params_only = downsample_params_corrected
    post_mlp_params = []

    if num_post_linear_layers > 0:
        conv_params_only = conv_params_only[:-num_post_linear_layers]
        downsample_params_only = downsample_params_only[:-num_post_linear_layers]
        post_mlp_params = regular_params_corrected[-num_post_linear_layers:]

    all_tcn_params = chunk_list(list(zip(conv_params_only, downsample_params_only)), 2)

    return all_tcn_params + post_mlp_params


def mask_non_subsection_weights(weight: np.ndarray, pe_rows: int, subsection_size: int):
    output_channels, input_channels = weight.shape[:2]

    for o in range(output_channels // pe_rows):
        # Can omit last axis since for a linear weight it does not exist
        # and since it does not change the result of the zeroing
        weight[o*pe_rows+subsection_size:(o+1)*pe_rows, :] = 0

    for i in range(input_channels // pe_rows):
        weight[:, i*pe_rows+subsection_size:(i+1)*pe_rows] = 0

    return weight


def get_random_tcn(input_blocks: int,
                   conv_blocks: List[int],
                   conv_kernel_sizes: List[int],
                   linear_blocks: Optional[List[int]] = None,
                   pe_rows: int = 16, weight_bit_width: int = 4,
                   act_bit_width: int = 4, bias_bit_width: int = 14,
                   scale_bit_width: int = 4, subsection_size: int = -1,
                   force_downsample: bool = False, slog2_weights: bool = True):
    assert len(conv_blocks) % 2 == 0, "The number of conv blocks must be even"
    assert len(conv_blocks) == len(conv_kernel_sizes), "The number of conv blocks and kernel sizes must be the same"

    if linear_blocks is None:
        linear_blocks = []
    
    all_blocks = conv_blocks + linear_blocks
    weight_shapes = list(zip([input_blocks] + all_blocks, all_blocks, conv_kernel_sizes + [-1]*len(linear_blocks)))

    if subsection_size != -1:
        pe_rows = subsection_size

    max_weight_value = 2**weight_bit_width

    weights = []
    downsample_params = []
    biases = []

    for input_channels, output_channels, kernel_size in weight_shapes:
        shape = [pe_rows*output_channels, pe_rows*input_channels]

        if kernel_size != -1:
            shape.append(kernel_size)
            
        weight = np.random.randint(0, max_weight_value, shape)
        # Create zero biases initially as we later adjust the biases
        # to be in the same range as the output of the layers
        bias = np.zeros(pe_rows*output_channels, dtype=int)

        weights.append(weight)
        biases.append(bias)

    weight_shapes_without_linear = weight_shapes

    if len(linear_blocks) > 0:
        weight_shapes_without_linear = weight_shapes[:-len(linear_blocks)]

    # Split in chunks of two since every conv layer has two weights
    conv_blocks_dims = chunk_list(weight_shapes_without_linear, 2)

    for conv_block in conv_blocks_dims:
        in_channels, _, _ = conv_block[0]
        _, out_channels, kernel_size = conv_block[1]

        downsample_params.append((None, None))

        if force_downsample or in_channels != out_channels:
            identity_weight = np.random.randint(0, max_weight_value, (pe_rows*out_channels, pe_rows*in_channels, 1))

            downsample_scale = random.randint(0, 2**(scale_bit_width-2)-1)
            downsample_params.append((identity_weight, downsample_scale))
        else:
            residual_scale = random.randint(0, 2**(scale_bit_width-2)-1)
            downsample_params.append((None, residual_scale))

    # Next, with the randomly initialized weights, we compute the output of the network at
    # each layer to determine scales that avoid overflow in all cases and the biases that
    # are in the same range as the output of the layers
    scales = [0] * len(all_blocks)

    input_length = get_receptive_field_size(conv_kernel_sizes[0::2], len(conv_blocks) // 2, 2)
    max_input_tensor = np.full((pe_rows*input_blocks, input_length), max_weight_value - 1)

    layer_idx = 0

    while layer_idx < len(all_blocks):
        regular_params = list(zip(weights, biases, scales))
        mlp_params = [] if len(linear_blocks) == 0 else regular_params[-len(linear_blocks):]
        all_tcn_params = chunk_list(list(zip(regular_params, downsample_params)), 2) + mlp_params
        output, intermediates, _ = tcn_network(max_input_tensor, all_tcn_params, weight_bit_width=weight_bit_width, act_bit_width=-1, accum_bit_width=-1, slog2_weights=slog2_weights)
        intermediates.append(output)

        current_step_out = intermediates[layer_idx]

        # Compute a custom lower bound on the biases to make sure that they do not dominate the result
        bias_bound = min(np.abs(current_step_out).max() // 2, 2**(bias_bit_width-1))

        # When testing subsection mode, it is more easily possible that ReLU is activated for
        # every output neuron, meaning that the bias_bound is zero
        if bias_bound == 0:
            bias_bound = 2**(bias_bit_width-1) // 8

        bias = np.random.randint(-bias_bound, bias_bound, biases[layer_idx].shape)
        biases[layer_idx] = bias

        # current_step_out is of shape (out_channels, time_steps) while
        # bias is of shape (out_channels,) so we need transpose twice
        # to add the bias to all timestep outputs
        current_step_out = (current_step_out.T + bias).T

        max_act_value = np.max(current_step_out, 0).max()

        if max_act_value < 2**act_bit_width:
            scale = 0
        else:
            # Compute scale based on maximum positive output value to avoid overflow
            scale = clog2(max_act_value) - act_bit_width + 1

        assert scale >= 0, "Scale must be greater than or equal to 0"
        assert scale < 2**scale_bit_width, "Scale must be less than 2^scale_bit_width"

        scales[layer_idx] = scale
        layer_idx += 1
    
    regular_params = list(zip(weights, biases, scales))
    mlp_params = [] if len(linear_blocks) == 0 else regular_params[-len(linear_blocks):]
    all_tcn_params = chunk_list(list(zip(regular_params, downsample_params)), 2) + mlp_params

    return all_tcn_params


def get_random_input_tensor_by_length(input_blocks: int, input_length: int, act_bit_width: int = 4, pe_rows: int = 16, subsection_size: int = 4):
    actual_rows = subsection_size if subsection_size != -1 else pe_rows
    input_tensor = np.random.randint(0, 2**act_bit_width, (actual_rows*input_blocks, input_length), dtype=np.int32)

    return input_tensor


def get_random_input_tensor(input_blocks: int, conv_blocks: List[int], kernel_sizes: Union[int, List[int]], act_bit_width: int = 4, pe_rows: int = 16, subsection_size: int = 4):
    input_length = get_receptive_field_size(kernel_sizes, len(conv_blocks) // 2, 2)

    return get_random_input_tensor_by_length(input_blocks, input_length, act_bit_width, pe_rows, subsection_size)


def get_used_bias_bit_width(layers: QuantLayers):
    min_bias_value = 0
    max_bias_value = 0

    for layer in layers:
        # If we encounter a linear layer
        if len(layer) == 3:
            _, bias, _ = layer

            min_bias_value = min(min_bias_value, bias.min())
            max_bias_value = max(max_bias_value, bias.max())
        elif len(layer) == 2:
            ((_, bias1, _), _), ((_, bias2, _), _) = layer

            min_bias_value = min(min_bias_value, bias1.min(), bias2.min())
            max_bias_value = max(max_bias_value, bias1.max(), bias2.max())
        else:
            raise ValueError("Invalid layer configuration")

    min_bit_width = clog2(abs(min_bias_value))

    if min_bias_value < 0:
        min_bit_width += 1

    max_bit_width = clog2(abs(max_bias_value))

    if max_bias_value < 0:
        max_bit_width += 1

    return max(min_bit_width, max_bit_width)


def get_all_weights_and_biases(layers: QuantLayers):
    weights = []
    biases = []

    for layer in layers:
        # If we encounter a linear layer
        if len(layer) == 3:
            weight, bias, _ = layer

            weights.append(weight.flatten().tolist())
            biases.append(bias.flatten().tolist())
        elif len(layer) == 2:
            ((weight1, bias1, scale1), _), ((weight2, bias2, scale2), (downsample_weight, downsample_scale)) = layer

            weights.append(weight1.flatten().tolist())
            weights.append(weight2.flatten().tolist())

            biases.append(bias1.flatten().tolist())
            biases.append(bias2.flatten().tolist())

            if downsample_weight is not None:
                weights.append(downsample_weight.flatten().tolist())
        else:
            raise ValueError("Invalid layer configuration")

    return weights, biases


def get_output_size(layers: QuantLayers):
    layer = layers[-1]

    if len(layer) == 3:
        weight, _, _ = layer

        return weight.shape[0]
    elif len(layer) == 2:
        _, ((weight2, _, _), _) = layer

        return weight2.shape[0]
    else:
        raise ValueError("Invalid layer configuration")


def get_kernel_sizes_and_num_tcn_conv_layers(layers: QuantLayers):
    num_tcn_conv_layers = 0
    kernel_sizes = []

    for layer in layers:
        # If we encounter a linear layer
        if len(layer) == 3:
            pass
        elif len(layer) == 2:
            ((weight1, _, _), _), ((weight2, _, _), _) = layer

            assert weight1.shape[2] == weight2.shape[2], "Only the same kernel size in both conv layers of a TCN layer is supported for now"

            kernel_sizes.append(weight1.shape[2])
            num_tcn_conv_layers += 1
        else:
            raise ValueError("Invalid layer configuration")

    return kernel_sizes, num_tcn_conv_layers


def get_quant_input(x: np.ndarray, quant_layers: QuantLayers, quant_in: Dict, clip: bool = True, pad: str = 'pre'):
    kernel_sizes, num_tcn_conv_layers = get_kernel_sizes_and_num_tcn_conv_layers(quant_layers)

    rf = get_receptive_field_size(kernel_sizes, num_tcn_conv_layers)

    if len(x.shape) == 1:
        x = x[np.newaxis, ...]

    if pad == 'post':
        x = float_to_uint(x, **quant_in, clip=clip)

    # If it is not bigger than one, the network is an MLP
    if rf > 1:
        # Pad before converting to integer to make sure that the zero padding
        # can optionally be quantized into zero_point values if required

        if x.shape[1] < rf:
            x = np.pad(x, ((0, 0), (rf-x.shape[1], 0)), mode='constant')
        
    if pad == 'pre':
        x = float_to_uint(x, **quant_in, clip=clip)

    return x
