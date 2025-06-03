from typing import OrderedDict, Union, Optional, List
from pathlib import Path

import numpy as np

from tqdm import trange, tqdm

from chameleon.core.shared_utils import QuantLayers
from chameleon.core.numpy.tcn import fc_bias_relu, tcn_network
from chameleon.core.net_load_utils import get_quant_input
from chameleon.core.net_transfer_utils import get_few_shot_scales, get_quant_state_dict_and_layers

from chameleon.core.learning_utils import (
    left_right_shift,
    compute_expected_weight_and_bias,
)


def learn_with_few_shots(
    net_path: Union[str, Path],
    iterable_dataset: iter,
    nearest_neighbor: bool = False,
    use_l2_distance: bool = True,
    shots: int = 5,
    ways: int = 5,
    query_shots: int = 15,
    num_repeats: int = 20,
    act_bit_width: int = 4,
    weight_bit_width: int = 4,
    accum_bit_width: int = 18,
    scale_bit_width: int = 4,
    accepted_layers: Optional[List[str]] = None,
    n_last_layers_to_remove: Optional[int] = None,
    clip: bool = True,
    pad: str = "pre",
    max_shots: int = 127
):
    in_quant, quant_layers = get_quant_state_dict_and_layers(
        net_path,
        scale_bit_width=scale_bit_width,
        accepted_layers=accepted_layers,
        n_last_layers_to_remove=n_last_layers_to_remove,
    )

    few_shot_scale, (_, k_shot_division_scale) = get_few_shot_scales(
        shots, weight_bit_width, act_bit_width, max_shots
    )

    few_shot_weight_scale = few_shot_scale
    few_shot_bias_scale = -few_shot_scale - k_shot_division_scale

    accs = []

    if nearest_neighbor:
        ways = shots * ways
        shots = 1

    for _ in trange(num_repeats, desc="Repeats"):
        ((X_support, y_support), (X_query, y_query)) = next(iterable_dataset)

        ways_embeds = []

        for way in trange(ways, desc="Training ways", leave=False):
            embds = []

            for i in range(shots):
                sample_idx = way * shots + i

                assert (
                    y_support[sample_idx] == way
                ), "Support set is not correctly labeled"

                x = get_quant_input(X_support[sample_idx], quant_layers, in_quant, clip=clip, pad=pad)

                emb, _, _ = tcn_network(
                    x,
                    quant_layers,
                    weight_bit_width=weight_bit_width,
                    act_bit_width=act_bit_width,
                    accum_bit_width=accum_bit_width,
                    slog2_weights=True,
                )

                embds.append(emb)

            embds = np.array(embds)
            sum_embds = np.sum(embds, axis=0)
            shift_sum_embds = left_right_shift(sum_embds, few_shot_weight_scale)

            assert np.max(shift_sum_embds) <= 2 ** (
                2 ** (weight_bit_width - 1) - 1
            ), f"Embeddings are too large ({np.max(shift_sum_embds)} @ {way*shots+i}) for the given bit width. Decrease the few_shot_scale value."

            shift_sum_embds = np.where(shift_sum_embds == 0, 1, shift_sum_embds)
            log2s = np.log2(shift_sum_embds)
            flog2_sum_embds = np.floor(log2s)
            correct_embeds = flog2_sum_embds.flatten().astype(int)
            ways_embeds.append(correct_embeds)

        num_correct = 0

        ways_arr, bias = compute_expected_weight_and_bias(
            ways_embeds, weight_bit_width, few_shot_bias_scale, use_l2_distance
        )

        for i, (X_query_sample, y_query_sample) in tqdm(
            enumerate(zip(X_query, y_query)),
            "Testing query samples",
            total=query_shots * ways,
            leave=False
        ):
            x = get_quant_input(X_query_sample, quant_layers, in_quant, clip=clip, pad=pad)

            correct_emb_out, _, _ = tcn_network(
                x,
                quant_layers,
                act_bit_width=act_bit_width,
                accum_bit_width=accum_bit_width,
                slog2_weights=True,
            )

            _, correct_out = fc_bias_relu(
                correct_emb_out, ways_arr, bias, -1, -1, 2 ** (accum_bit_width - 1)
            )

            correct_out_argmax = np.argmax(correct_out)

            if nearest_neighbor:
                correct_out_argmax = correct_out_argmax // shots

            num_correct += correct_out_argmax == y_query_sample

        accuracy = num_correct / len(y_query)

        accs.append(accuracy.item())

    return accs


def learn_in_context(
    quant_layers: QuantLayers,
    quant_state_dict: OrderedDict,
    iterable_dataset: iter,
    few_shot_scale: int,
    nearest_neighbor: bool = False,
    use_l2_distance: bool = True,
    shots: int = 5,
    ways: int = 5,
    query_shots: int = 15,
    num_repeats: int = 20,
    weight_bit_width: int = 4,
    act_bit_width: int = 4,
    accum_bit_width: int = 18,
):
    pass
