import yaml
import importlib.resources

import cocotb

from autolightning import auto_data


from .utils import learn_with_few_shots


CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"


with importlib.resources.files("metalarena").joinpath("configs/omniglot/proto.yaml").open("r") as f:
    data_args = yaml.safe_load(f)

ds = auto_data(data_args)
ds.setup('validate')
ds_val = ds.get_transformed_dataset('val').dataset


@cocotb.test()
async def test_120k_tcn_5_way_1_shot_omniglot(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        ds_val, 2, 1, 5, 1, 0, CLOCK_FREQ,
        (None, (0.959, 0.986)), "../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl",
        n_last_layers_to_remove=0, require_single_chunk=True, seed=0, l2_options=True,
        check_memory_contents=True
    )


@cocotb.test()
async def test_120k_tcn_5_way_5_shot_omniglot(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        ds_val, 5, 10, 5, 3, 0, CLOCK_FREQ,
        (None, (0.99, 0.99)), "../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl",
        n_last_layers_to_remove=0, require_single_chunk=True, seed=0, l2_options=True,
        check_memory_contents=True
    )


@cocotb.test()
async def test_120k_tcn_20_way_1_shot_omniglot(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        ds_val, 1, 3, 20, 3, 0, CLOCK_FREQ,
        (None, (0.799, 0.833)), "../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl",
        n_last_layers_to_remove=0, require_single_chunk=True, seed=0, l2_options=True,
        check_memory_contents=True
    )


@cocotb.test()
async def test_120k_tcn_2_way_continued_17_way_1_shot_omniglot(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        ds_val, 1, 3, 17, 3, 2, CLOCK_FREQ,
        (None, (0.784, 0.829)), "../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl",
        n_last_layers_to_remove=0, require_single_chunk=True, seed=0, l2_options=True,
        check_memory_contents=True
    )
