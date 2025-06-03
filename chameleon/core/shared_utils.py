from typing import List, Tuple, Union
import math
import os
import random

import numpy as np

from torch.utils.data import Dataset


QuantLayer = List[Tuple[Union[Tuple[np.ndarray, np.ndarray, int], Union[Tuple[None, None], Tuple[None, int], Tuple[np.ndarray, int]]], Tuple[np.ndarray, np.ndarray, int]]]
QuantLayers = List[QuantLayer]


def clog2(x: int):
    """Return the ceiling of the base-2 logarithm of x.

    Args:
        x (int): The number to take the logarithm of.

    Returns:
        int: The ceiling of the base-2 logarithm of x.
    """

    return math.ceil(math.log2(x))


def iclog2(x: int):
    return 2**clog2(x)


def flog2(x: int):
    """Return the floor of the base-2 logarithm of x.

    Args:
        x (int): The number to take the logarithm of.

    Returns:
        int: The floor of the base-2 logarithm of x.
    """

    return math.floor(math.log2(x))


def iflog2(x: int):
    return 2**flog2(x)


def twos_complement_to_int(x: int, bits: int):
    """compute the 2's complement of int value val"""
    if (x & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        x = x - (1 << bits)        # compute negative value
    return x                         # return positive value as is


def correct_subsection_mode_argmax(x: int, pe_rows: int = 16, subsection_size: int = 3):
    return x % subsection_size + (x // pe_rows) * subsection_size


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def assert_asic_out(asic_out, correct_out: np.ndarray, unscaled: np.ndarray, classification: bool, is_new_task: bool, send_all_argmax_chunks: bool = False):
    if is_new_task:
        assert asic_out is None, f"ASIC output {asic_out} should be None for new task"
    elif classification:
        argmax = np.argmax(unscaled)

        if send_all_argmax_chunks:
            asic_out, asic_max = asic_out
            assert asic_max == unscaled[argmax], f"ASIC max {asic_max} does not match expected max {unscaled[argmax]}"

        assert asic_out == argmax, f"ASIC argmax {asic_out} does not match expected argmax {argmax}"
    else:
        assert np.array_equal(asic_out, correct_out)


class NpyDataset(Dataset):
    def __init__(self, directory_path):
        """
        Args:
            directory_path (str): Path to the directory containing <name>.npy files.
        """

        self.directory_path = directory_path
        self.file_names = sorted([f for f in os.listdir(directory_path) if f.endswith('.npy')])

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory_path, self.file_names[idx])
        data = np.load(file_path)
        return data

    def __len__(self):
        return len(self.file_names)
