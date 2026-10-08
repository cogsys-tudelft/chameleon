"""Convert the Liberty files of the IHP SG13G2 SRAM macros to the slew convention of the standard cells.

The SRAM Liberty files measure transitions between 30% and 70% with slew_derate_from_library 0.5,
the standard cells between 20% and 80% with a derate of 1. OpenSTA has to convert slews between
the two, and on a net that drives several pins of the same SRAM macro, it applies the conversion
(x4/3) once more for every further pin. The slews on such nets grow to thousands of ns, and the
extrapolated setup times of the SRAM pins give a setup WNS of thousands of ns.

This script rewrites the SRAM Liberty files in the convention of the standard cells, so that no
conversion is needed: the thresholds become 20%/80% with a derate of 1, and every transition value
(table indices of transition variables, output transition tables and max_transition) is multiplied
by 0.75. For a linear ramp, a 30-70% time of 0.5 * L equals a 20-80% time of 0.75 * L, so the
patched files describe the same timing. Capacitances, delays, setup/hold constraints and power
are not changed.

Usage (from the code directory):
    python ihp_sg13g2/patch_sram_lib.py

The patched files are written to ihp_sg13g2/sram_lib, which ihp_sg13g2/config.yaml uses.
"""

import argparse
import os
import re
from pathlib import Path

FLOW_DIR = Path(__file__).resolve().parent
SRAM_MACROS = [
    "RM_IHPSG13_1P_256x32_c2_bm_bist",
    "RM_IHPSG13_1P_512x64_c2_bm_bist",
    "RM_IHPSG13_2P_256x32_c2_bm_bist",
]

# 20-80% time of the ramp whose 30-70% time is slew_derate_from_library (0.5) * L
SLEW_SCALE = 0.75
NUMBER = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"
TABLE_GROUPS = ("rise_constraint", "fall_constraint", "cell_rise", "cell_fall", "rise_transition",
                "fall_transition", "rise_power", "fall_power")


def scale_numbers(text: str) -> str:
    return re.sub(NUMBER, lambda m: f"{float(m.group(0)) * SLEW_SCALE:.6g}", text)


def set_attribute(text: str, name: str, value: str) -> str:
    text, count = re.subn(rf"({name}\s*:\s*)[^;]*;", rf"\g<1>{value} ;", text)

    if count == 0:
        raise ValueError(f"Attribute {name} not found")

    return text


def patch_liberty(text: str) -> str:
    if not re.search(r"slew_derate_from_library\s*:\s*0\.5\s*;", text):
        raise ValueError("Expected slew_derate_from_library : 0.5 (already patched?)")

    for edge in ("rise", "fall"):
        text = set_attribute(text, f"slew_lower_threshold_pct_{edge}", "20")
        text = set_attribute(text, f"slew_upper_threshold_pct_{edge}", "80")
    text = set_attribute(text, "slew_derate_from_library", "1.0")

    # For each table template, which of its variables are transitions
    transition_variables = {}
    for name, body in re.findall(r"lu_table_template\s*\(\s*(\w+)\s*\)\s*\{(.*?)\}", text, re.S):
        transition_variables[name] = {
            int(index) for index, variable in re.findall(r"variable_(\d)\s*:\s*(\w+)", body) if "transition" in variable
        }

    def patch_table(match: re.Match) -> str:
        group, template, body = match.groups()

        for index in transition_variables.get(template, set()):
            body = re.sub(rf'(index_{index}\s*\(\s*")([^"]*)(")',
                          lambda m: m.group(1) + scale_numbers(m.group(2)) + m.group(3), body)

        # Output transition tables hold transitions themselves
        if group in ("rise_transition", "fall_transition"):
            body = re.sub(r"(values\s*\()(.*?)(\)\s*;)",
                          lambda m: m.group(1) + scale_numbers(m.group(2)) + m.group(3), body, flags=re.S)

        return f"{group}({template}) {{{body}}}"

    text = re.sub(rf'\b({"|".join(TABLE_GROUPS)})\s*\(\s*"?(\w+)"?\s*\)\s*\{{(.*?)\}}', patch_table, text, flags=re.S)

    return re.sub(rf'((?:default_)?max_transition\s*:\s*"?)({NUMBER})("?\s*;)',
                  lambda m: m.group(1) + scale_numbers(m.group(2)) + m.group(3), text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=FLOW_DIR / "sram_lib")
    args = parser.parse_args()

    lib_dir = Path(os.environ.get("PDK_ROOT", "~/.ciel")).expanduser() / "ihp-sg13g2" / "libs.ref" / "sg13g2_sram" / "lib"
    args.output.mkdir(parents=True, exist_ok=True)

    for macro in SRAM_MACROS:
        libs = sorted(lib_dir.glob(f"{macro}_*.lib"))

        if len(libs) == 0:
            raise FileNotFoundError(f"No Liberty files for {macro} in {lib_dir} (is PDK_ROOT set and ihp-sg13g2 enabled?)")

        for lib in libs:
            (args.output / lib.name).write_text(patch_liberty(lib.read_text()))
            print(f"Patched {lib.name}")


if __name__ == "__main__":
    main()
