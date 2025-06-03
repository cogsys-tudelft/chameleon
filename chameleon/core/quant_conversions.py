from typing import Optional

import numpy as np


def int_to_slog2(x: np.ndarray, bit_width: int = 4) -> np.ndarray:
    """Convert an array of signed integers to their corresponding slog2 format.
    
    For example `int_to_slog2(np.array([-128]), 4)` -> `15`

    Args:
        x (np.ndarray): Input array
        bit_width (int, optional): slog2 format bit width. Defaults to 4.
    """

    int_log2_weights = np.log2(np.abs(x))

    # For each negative weight, make it positive and add 2**(bit_width-1)
    int_log2_weights = int_log2_weights + (np.sign(x) == -1) * (2**(bit_width-1))
    int_log2_weights = int_log2_weights.astype(int)

    assert np.abs(int_log2_weights).max() < 2**(bit_width), "Values are too large for the given bit width"
    assert np.abs(int_log2_weights).min() >= 0, "Values are too small for the given bit width"

    return int_log2_weights


def slog2_to_int(x: np.ndarray, bit_width: int = 4) -> np.ndarray:
    """Convert an array of slog2 formatted values to their corresponding signed integer values.
    
    For example `slog2_to_int(np.array([15]), 4)` -> `-128`

    Args:
        x (np.ndarray): Input array
        bit_width (int, optional): slog2 format bit width. Defaults to 4.
    """

    assert np.abs(x).max() < 2**(bit_width), "Values are too large for the given bit width"
    assert x.min() >= 0, "Values are too small for the given bit width"

    is_negative = x >= 2**(bit_width-1)
    signs = -((is_negative)*2-1)
    powers_of_two = x - is_negative * 2**(bit_width-1)
    log2_weights = 2**powers_of_two * signs

    return log2_weights


def float_to_slog2(x: np.ndarray, scale: float, bit_width: int = 4):
    # Give +1 extra bit for the to int conversion, to make sure that weights
    # such as -128 and 128 are not de-asserted. Asserts in int_to_slog2 will
    # make sure the final values are still within the bounds of the bit width.
    int_values = float_to_int(x, scale, bit_width=2**(bit_width-1)+1, signed=True)

    return int_to_slog2(int_values, bit_width)


def float_to_int(x: np.ndarray, scale: float, zero_point: Optional[float] = None, bit_width: int = 4, signed: bool = True, float_to_int_impl: str = 'round', clip: bool = False) -> np.ndarray:
    values = x / scale
    
    if zero_point:
        values = values + zero_point

    rounded_values = getattr(np, float_to_int_impl)(values)

    int_values = rounded_values.astype(int)

    if signed:
        min = -2**(bit_width-1)
        max = 2**(bit_width-1) - 1
    else:
        min = 0
        max = 2**bit_width - 1

    if clip:
        int_values = np.clip(int_values, min, max)

    assert int_values.min() >= min, f"Values are too small for the given bit width: {int_values.min()}"
    assert int_values.max() <= max, f"Values are too large for the given bit width: {int_values.max()}"

    return int_values


def float_to_uint(x: np.ndarray, scale: float, zero_point: Optional[float] = None, bit_width: int = 4, float_to_int_impl: str = 'round', clip: bool = True) -> np.ndarray:
    return float_to_int(x, scale, zero_point, bit_width, signed=False, float_to_int_impl=float_to_int_impl, clip=clip)
