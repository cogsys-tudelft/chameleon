"""Turn LibreLane's post-synthesis netlist into a netlist that Verilator can simulate.

The IHP SG13G2 standard cell Verilog models use UDPs, which Verilator does not
support. Instead, Yosys imports the cell functions from the Liberty file and the
netlist is flattened into plain Verilog. The SRAM macros are kept as instances and
are simulated with the functional models from the PDK.

Usage (from the code directory, with Yosys on the PATH):
    python ihp_sg13g2/make_sim_netlist.py [path/to/chameleon.nl.v]

Without an argument, the netlist of the latest LibreLane run in ihp_sg13g2/runs is used.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

FLOW_DIR = Path(__file__).resolve().parent
TOP = "chameleon"
SRAM_MACROS = [
    "RM_IHPSG13_1P_256x32_c2_bm_bist",
    "RM_IHPSG13_1P_512x64_c2_bm_bist",
    "RM_IHPSG13_2P_256x32_c2_bm_bist",
]


def latest_synthesis_netlist() -> Path:
    netlists = sorted((FLOW_DIR / "runs").glob(f"*/*-yosys-synthesis/{TOP}.nl.v"), key=lambda p: p.stat().st_mtime)

    if len(netlists) == 0:
        raise FileNotFoundError("No synthesized netlist found, run LibreLane first (see config.yaml)")

    return netlists[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("netlist", nargs="?", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=FLOW_DIR / "netlist" / f"{TOP}.sim.v")
    args = parser.parse_args()

    netlist = args.netlist or latest_synthesis_netlist()
    pdk_dir = Path(os.environ.get("PDK_ROOT", "~/.ciel")).expanduser() / "ihp-sg13g2" / "libs.ref"
    stdcell_lib = pdk_dir / "sg13g2_stdcell" / "lib" / "sg13g2_stdcell_typ_1p20V_25C.lib"
    sram_libs = [pdk_dir / "sg13g2_sram" / "lib" / f"{macro}_typ_1p20V_25C.lib" for macro in SRAM_MACROS]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    script = "\n".join([
        # Standard cells become modules with their Liberty function (clock gating
        # cells have no function and are not used by the synthesized netlist)
        f"read_liberty -ignore_miss_func {stdcell_lib}",
        # SRAM macros stay black boxes
        *[f"read_liberty -lib {lib}" for lib in sram_libs],
        f"read_verilog {netlist}",
        f"hierarchy -check -top {TOP}",
        "flatten",
        "opt_clean -purge",
        f"write_verilog -noattr {args.output}",
        f"stat -top {TOP}",
    ])

    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as f:
        f.write(script)
        script_path = f.name

    print(f"Converting {netlist} -> {args.output}")
    subprocess.check_call(["yosys", "-q", "-l", str(args.output.with_suffix(".log")), "-s", script_path])
    os.unlink(script_path)


if __name__ == "__main__":
    main()
