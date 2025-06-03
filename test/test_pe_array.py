from utils import run_module_test, cli

def test_pe_array(simulator: str, parameters=None):
    run_module_test("pe_array",
                    parameters = parameters,
                    include_src_dir=True,
                    extension="sv",
                    defines={"FLAT_WEIGHTS": 1},
                    simulator=simulator)

if __name__ == "__main__":
    test_pe_array(cli()[0], {"BIAS_BIT_WIDTH": "14", "ACCUMULATION_BIT_WIDTH": "18", "WEIGHT_BIT_WIDTH": "4"})
