import numpy as np

import cocotb
from cocotb.triggers import Timer

from chameleon.core.shared_utils import twos_complement_to_int
from chameleon.core.quant_conversions import slog2_to_int


@cocotb.test()
async def test_all_input_combinations(dut):
    WEIGHT_BIT_WIDTH = int(dut.WEIGHT_BIT_WIDTH.value)

    for in_val in range(2**int(dut.INPUT_BIT_WIDTH.value)):
        for weight in range(2**WEIGHT_BIT_WIDTH):
            getattr(dut, "in").value = in_val
            dut.weight.value = weight
            expected = slog2_to_int(np.array([weight]), WEIGHT_BIT_WIDTH)[0] * in_val

            await Timer(1, unit='ns')

            actual = twos_complement_to_int(int(dut.out.value), len(dut.out.value))

            assert actual == expected, f"Expected {expected} from {in_val} * {weight}, got {actual}"
