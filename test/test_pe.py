import itertools

import pytest

from utils import run_module_test, cli

MAX_WEIGHT_BIT_WIDTH = 16
MAX_INPUT_BIT_WIDTH = 16

# Generate all possible pairs of input and weight bit widths
input_weight_bit_widths = list(itertools.product(range(1, MAX_INPUT_BIT_WIDTH + 1), range(1, MAX_WEIGHT_BIT_WIDTH + 1)))

# Give them names
input_weight_bit_widths = [{"WEIGHT_BIT_WIDTH": str(weight), "INPUT_BIT_WIDTH": str(input)} for input, weight in input_weight_bit_widths]

@pytest.mark.parametrize("parameters", input_weight_bit_widths)
def test_pe(simulator: str, parameters):
    run_module_test("pe",
                    parameters = parameters,
                    extension="v",
                    simulator=simulator)

if __name__ == "__main__":
    test_pe(cli()[0], {"WEIGHT_BIT_WIDTH": "4", "INPUT_BIT_WIDTH": "4"})
