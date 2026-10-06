"""System tests for the IHP SG13G2 open-source flow.

Runs a representative subset of the Chameleon system tests in three modes:

- ``behavioral``: RTL with the behavioral SRAM models from asic-cells (reference)
- ``ihp``: RTL with the SRAMs mapped to IHP SG13G2 SRAM macros (src/ihp_sg13g2)
- ``netlist``: post-synthesis gate-level netlist from LibreLane (see ihp_sg13g2/)

The IHP SRAM macro models are taken from the PDK, so PDK_ROOT must point to a
Ciel PDK root with ihp-sg13g2 enabled (default: ~/.ciel).
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import pytest

from utils import run_module_test

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent

# Subset of the system tests that covers MLPs, TCNs, end-to-end MNIST
# classification (incl. 4x4 subsection mode) and few-shot learning
SYSTEM_TEST_MODULES = [
    "tests.chameleon.fc_mlp",
    "tests.chameleon.tcn",
    "tests.chameleon.mnist",
    "tests.chameleon.few_shot_mnist",
]
SYSTEM_TESTS = [
    "single_block_mlp_2x1",
    "mlp_with_four_layers",
    "single_tcn_layer",
    "triple_tcn_layer_with_mlp",
    "test_14k4_tcn_mnist",
    "test_70k_tcn_mnist",
    "test_70k_tcn_5_way_1_shot_mnist",
]

# Macros used by src/ihp_sg13g2 for the default Chameleon parameters
IHP_SRAM_MACROS = [
    "RM_IHPSG13_1P_256x32_c2_bm_bist",
    "RM_IHPSG13_1P_512x64_c2_bm_bist",
    "RM_IHPSG13_2P_256x32_c2_bm_bist",
]

ASIC_CELLS_DIRS = ["aer", "clock", "clock_domain_crossing", "spi_interface", "pipeline"]

VERILATOR_COMPILE_ARGS = ['-Wno-UNOPTFLAT', '-Wno-WIDTHTRUNC']
# The IHP dual-port SRAM model writes its memory array from both ports' always blocks
IHP_VERILATOR_COMPILE_ARGS = VERILATOR_COMPILE_ARGS + ['-Wno-MULTIDRIVEN']


def ihp_pdk_dir() -> Path:
    return Path(os.environ.get("PDK_ROOT", "~/.ciel")).expanduser() / "ihp-sg13g2"


def ihp_sram_model_sources() -> List[str]:
    verilog_dir = ihp_pdk_dir() / "libs.ref" / "sg13g2_sram" / "verilog"

    sources = [
        verilog_dir / "RM_IHPSG13_1P_core_behavioral_bm_bist.v",
        verilog_dir / "RM_IHPSG13_2P_core_behavioral_bm_bist_ideal.v",
    ] + [verilog_dir / f"{macro}.v" for macro in IHP_SRAM_MACROS]

    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f"IHP SRAM model not found: {source} (is PDK_ROOT set and ihp-sg13g2 enabled?)")

    return [str(source) for source in sources]


def run_system_tests(mode: str, sim_build: str, tests: Optional[List[str]] = None, waves: bool = False):
    tests = tests or SYSTEM_TESTS
    os.environ["TESTCASE"] = ",".join(tests)

    include_dirs = [f"../deps/asic-cells/src/{d}" for d in ASIC_CELLS_DIRS]
    include_dirs.append("../deps/verilog-array-operations/src")

    if mode == "behavioral":
        include_dirs.append("../deps/asic-cells/src/sram")

        return run_module_test("chameleon",
                               include_src_dir=True,
                               extension="sv",
                               include_dirs=include_dirs,
                               compile_args=VERILATOR_COMPILE_ARGS,
                               module_path=",".join(SYSTEM_TEST_MODULES),
                               waves=waves,
                               sim_build=sim_build)
    elif mode == "ihp":
        include_dirs.append("../src/ihp_sg13g2")

        return run_module_test("chameleon",
                               include_src_dir=True,
                               extension="sv",
                               include_dirs=include_dirs,
                               compile_args=IHP_VERILATOR_COMPILE_ARGS,
                               defines={"FUNCTIONAL": 1},
                               verilog_sources=ihp_sram_model_sources(),
                               module_path=",".join(SYSTEM_TEST_MODULES),
                               waves=waves,
                               sim_build=sim_build)
    elif mode == "netlist":
        netlist = CODE_DIR / "ihp_sg13g2" / "netlist" / "chameleon.sim.v"

        if not netlist.exists():
            raise FileNotFoundError(f"Netlist {netlist} not found, run ihp_sg13g2/make_sim_netlist.py first")

        return run_module_test("chameleon_netlist_tb",
                               source_dir=str(CODE_DIR / "ihp_sg13g2"),
                               extension="sv",
                               verilog_sources=[str(netlist)] + ihp_sram_model_sources(),
                               compile_args=IHP_VERILATOR_COMPILE_ARGS + ['-Wno-fatal', '-Wno-lint', '-Wno-style'],
                               defines={"FUNCTIONAL": 1},
                               module_path=",".join(SYSTEM_TEST_MODULES),
                               waves=waves,
                               sim_build=sim_build)
    else:
        raise ValueError(f"Unknown mode: {mode}")


@pytest.mark.parametrize("mode", ["behavioral", "ihp"])
def test_chameleon_ihp(mode: str):
    run_system_tests(mode, f"sim_build_{mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["behavioral", "ihp", "netlist"])
    parser.add_argument("-t", "--tests", nargs="+", help="Tests to run (default: the system test subset)")
    parser.add_argument("-b", "--sim-build", type=str, default=None)
    parser.add_argument("-w", "--waves", action="store_true")
    args = parser.parse_args()

    run_system_tests(args.mode, args.sim_build or f"sim_build_{args.mode}", args.tests, args.waves)
