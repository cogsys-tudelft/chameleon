import itertools

import pytest

from utils import run_module_test, cli

MAX_WIDTH = 12
MAX_N_EXPONENT = 8

all_combs = list(itertools.product(range(1, MAX_WIDTH + 1), range(1, MAX_N_EXPONENT + 1)))
parameters = [{"WIDTH": str(width), "N": str(2**n_exp)} for width, n_exp in all_combs]


@pytest.mark.parametrize("parameters", parameters)
def test_argmax_tree(simulator: str, waves: bool, sim_build: str, simulation_args: list, parameters: dict):
    run_module_test("argmax_tree",
                    parameters=parameters,
                    include_src_dir=True,
                    waves=waves,
                    use_default_compile_args=False,
                    simulator=simulator,
                    simulation_args=simulation_args,
                    sim_build=sim_build)

if __name__ == "__main__":
    test_argmax_tree(*cli(), {"WIDTH": 15, "N": 16})
