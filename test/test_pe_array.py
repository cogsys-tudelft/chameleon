from utils import run_module_test, cli


def test_pe_array(simulator: str, waves: bool, sim_build: str, simulation_args: list, parameters: dict):
    run_module_test("pe_array",
                    parameters=parameters,
                    include_src_dir=True,
                    waves=waves,
                    extension="sv",
                    # Required since most sims do not support 2D array ports at the top level
                    defines={"FLAT_WEIGHTS": 1},
                    simulator=simulator,
                    simulation_args=simulation_args,
                    sim_build=sim_build)
                    

if __name__ == "__main__":
    test_pe_array(*cli(), {"BIAS_BIT_WIDTH": "14", "ACCUMULATION_BIT_WIDTH": "18", "WEIGHT_BIT_WIDTH": "4", "ROWS": 8, "COLS": 8, "CLAMP": 1})
