from typing import Optional

import os
import glob
from pathlib import Path

from cocotb_test.simulator import run


# Ignore WIDTHEXPAND warnings: https://verilator.org/guide/latest/warnings.html#cmdoption-arg-WIDTHEXPAND
# This is when you use 4 bits but you would need 5 bits, but this is in 99.99% of the cases as intended
BASIC_COMPILE_ARGS_DICT = {
    "verilator": ['-Wno-WIDTHEXPAND', '-Wno-WIDTHTRUNC'],  # TODO: '--x-assign unique', '--x-initial unique'
    "questa": ['-O5']
}


def extract_verilog_files(directory):
    """
    Extracts all .v, .sv and .vh files in the specified directory.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        list: A list of paths to .v, .sv and .vh files in the directory.
    """

    v_files = glob.glob(os.path.join(directory, '*.v'))
    sv_files = glob.glob(os.path.join(directory, '*.sv'))
    vh_files = glob.glob(os.path.join(directory, '*.vh'))
    
    return v_files + sv_files + vh_files


def run_module_test(module_name: str,
                    extension: str = "v",
                    file_name: Optional[str] = None,
                    parameters : Optional[dict] = None,
                    include_src_dir: bool = False,
                    include_dirs: Optional[list] = None,
                    use_basic_compile_args: bool = True,
                    compile_args: Optional[list] = None,
                    waves: bool = False,
                    defines: Optional[dict] = None,
                    simulator: str = "verilator",
                    module_path: Optional[str] = None,
                    source_dir: Optional[str] = None,
                    verilog_sources: Optional[list[str]] = None,
                    **kwargs):
    file_dir = Path(__file__).resolve().parent
    source_dir = str(file_dir / ".." / "src") if source_dir is None else source_dir

    extra_args = []

    all_include_dirs = include_dirs or []

    if include_src_dir:
        all_include_dirs.append(source_dir)

    all_include_dirs = [str(file_dir / include_dir) for include_dir in all_include_dirs]

    if file_name is None:
        file_name = f"{module_name}.{extension}"


    verilog_sources = verilog_sources or []
    verilog_sources.append(f"{source_dir}/{file_name}")

    if len(all_include_dirs) > 0:
        if not (simulator == "verilator" or simulator == "questa" or simulator == "icarus"):
            raise ValueError(f"Include source directory behavior is not tested for this ({simulator}) simulator")
        
        if simulator == "questa":
            for include_dir in all_include_dirs:
                verilog_sources.extend(extract_verilog_files(include_dir))

    if use_basic_compile_args:
        basic_compile_args = BASIC_COMPILE_ARGS_DICT.get(simulator, None)

        if basic_compile_args == None:
            raise ValueError(f"Cannot enable basic compile args for simulator ({simulator}) for which no basic compile args are defined")
    else:
        basic_compile_args = []

    defines_list = None

    if defines:
        defines_list = []

        for key, value in defines.items():
            defines_list.append(f"{key}={value}")

    # By default, Verilator generates an FST file but we want a VCD file
    if waves and simulator == "verilator":
        extra_args.append("--trace")

        waves = False

    compile_args = list(set(basic_compile_args + (compile_args or [])))

    return run(
        simulator=simulator,
        # Remove possible duplicate verilog source file of main module
        # by using list and set
        verilog_sources=list(set(verilog_sources)),
        toplevel=module_name,
        module=f"tests.{module_name}_tests" if module_path is None else module_path,
        parameters=parameters,
        compile_args=compile_args,
        extra_args=extra_args,
        includes=all_include_dirs,
        defines=defines_list,
        waves=waves,
        simulation_args=kwargs.get("simulations_args", None),
        sim_build=kwargs.get("sim_build", "sim_build")
    )


def cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", help="Simulator to use", type=str, default='verilator')
    parser.add_argument("-w", "--waves", help="Create waveform file", action='store_true')
    parser.add_argument("-b", "--sim-build", help="Sim build directory", type=str, default='sim_build')
    parser.add_argument("-sargs", "--simulation-args", help="Arguments to pass to the simulation", nargs='+')

    args = parser.parse_args()

    return args.simulator, args.waves, args.sim_build, args.simulation_args
