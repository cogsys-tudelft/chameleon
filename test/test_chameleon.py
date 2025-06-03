from typing import List, Optional

from utils import run_module_test, cli


def test_chameleon(simulator: str, waves: bool, sim_build: str, simulations_args: Optional[List[str]] = None, parameters=None):
    asic_cells_dirs = ["aer", "clock", "clock_domain_crossing", "spi_interface", "sram", "pipeline"]
    include_dirs = [f"../deps/asic-cells/src/{dir}" for dir in asic_cells_dirs]
    include_dirs.append("../deps/verilog-array-operations/src")

    compile_args = None

    if simulator == 'verilator':
        # Dont fail on UNOPTFLAT due to the fact that Verilator thinks there is a loop in the design, while there is not.
        # Also dont fail on SELRANGE that occurs due to the subsection code
        compile_args=['-Wno-UNOPTFLAT', '-Wno-WIDTHTRUNC']

    run_module_test("chameleon",
                    parameters=parameters,
                    include_src_dir=True,
                    extension="sv",
                    include_dirs=include_dirs,
                    waves=waves,
                    simulator=simulator,
                    compile_args=compile_args,
                    simulations_args=simulations_args,
                    sim_build=sim_build)


if __name__ == "__main__":
    # Make sure that MNIST can be tested on an MLP in the sim
    # (but not in the real chip)
    test_chameleon(*cli(), {"INPUT_ROWS": 784 // 16})
