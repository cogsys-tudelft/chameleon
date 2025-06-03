from typing import Optional, Union, List
from pathlib import Path

import numpy as np

from tqdm import tqdm

from chameleon.core.net_load_utils import get_quant_input
from chameleon.core.net_transfer_utils import get_quant_state_dict_and_layers
from chameleon.core.numpy.tcn import tcn_network


def infer(
    net_path: Union[str, Path],
    dataset,
    classification: bool = True,
    limit_samples: Optional[int] = None,
    weight_bit_width: int = 4,
    act_bit_width: int = 4,
    accum_bit_width: int = 18,
    scale_bit_width: int = 4,
    accepted_layers: Optional[List[str]] = None,
    n_last_layers_to_remove: Optional[int] = None,
    clip: bool = True,
    pad: str = "pre",
):
    count = 0
    embeds = []
    preds_and_targets = []

    in_quant, quant_layers = get_quant_state_dict_and_layers(
        net_path,
        scale_bit_width=scale_bit_width,
        accepted_layers=accepted_layers,
        n_last_layers_to_remove=n_last_layers_to_remove,
    )

    if not limit_samples:
        limit_samples = len(dataset)

    for i in tqdm(
        range(len(dataset)),
        desc="Performing inference over dataset",
        total=limit_samples,
    ):
        x, y = dataset[i]

        x = get_quant_input(x, quant_layers, in_quant, clip=clip, pad=pad)

        embed_out, _, accum_out = tcn_network(
            x,
            quant_layers,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            accum_bit_width=accum_bit_width,
            slog2_weights=True,
        )

        out = accum_out if classification else embed_out

        preds_and_targets.append((out, y))

        if classification:
            count += np.argmax(accum_out) == y
        else:
            embeds.append(embed_out)

        if i == limit_samples - 1:
            break

    if classification:
        final_out = count / (limit_samples if limit_samples else len(dataset))
    else:
        final_out = np.array(embeds)

    return final_out, preds_and_targets
