import itertools

import pytest

from utils import run_module_test, cli

MAX_WIDTH = 12
MAX_N_EXPONENT = 8

all_combs = list(itertools.product(range(1, MAX_WIDTH + 1), range(1, MAX_N_EXPONENT + 1)))
parameters = [{"WIDTH": str(width), "N": str(2**n_exp)} for width, n_exp in all_combs]

@pytest.mark.parametrize("parameters", parameters)
def test_argmax_tree(simulator: str, waves: bool, parameters=None):
    run_module_test("argmax_tree",
                    parameters=parameters,
                    include_src_dir=True,
                    waves=waves,
                    use_basic_compile_args=False,
                    simulator=simulator)

if __name__ == "__main__":
    test_argmax_tree(*cli()[:2], {"WIDTH": 15, "N": 16})
