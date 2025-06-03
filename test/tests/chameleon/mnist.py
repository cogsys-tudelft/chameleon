from typing import Optional, List

import cocotb

from torchvision import datasets, transforms
import torch.nn as nn

from .utils import get_params_and_controller, caller_name


CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"


async def test_mnist(dut, expected_accuracy: float, quant_state_dict_file_path: str, start_dim: int, require_single_chunk: bool = False, accepted_layers: Optional[List[str]] = None, in_subsection_mode: bool = False, skip_if_padded: bool = False):
    _, c = get_params_and_controller(dut, CFG_MEMORY_FILE, POINTER_FILE)

    if skip_if_padded and c.is_padded:
        cocotb.log.info("> Skipping test because the ASIC is padded")
        return

    await c.start_and_verify_asic(1, CLOCK_FREQ, reset_host=False, verify_memories=False)
    await c.write_quant_state_dict_to_asic(
        quant_state_dict_file_path,
        True, accepted_layers, padding_value=0,
        subsection_network=in_subsection_mode
    )

    # Normalize images and flatten them
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        nn.Flatten(start_dim=start_dim)
    ])

    mnist_val = datasets.MNIST('../../data', train=False, download=True, transform=mnist_transform)

    prev_accuracy = -1

    for fill_input_memory in (False, True):
        for power_down_memories_while_running in (False, True):
            cocotb.log.info(f"> Running with: \n - fill_input_memory: {fill_input_memory}\n - power_down_memories_while_running: {power_down_memories_while_running}")

            accuracy, _ = await c.classify_dataset(
                mnist_val,
                quant_state_dict_file_path,
                n_samples_to_test=11,
                write_state_dict_to_asic=False,
                configure_processing_setup=True,
                require_single_chunk=require_single_chunk,
                fill_input_memory=fill_input_memory,
                power_down_memories_while_running=power_down_memories_while_running,
                in_subsection_mode=in_subsection_mode,
                accepted_layers=accepted_layers,
                expected_min_accuracy=expected_accuracy
            )

            if prev_accuracy != -1:
                assert accuracy == prev_accuracy, f"Accuracy is not consistent: {accuracy} != {prev_accuracy}"

            prev_accuracy = accuracy

    with open("passed.txt", "a") as f:
        f.write(f"{caller_name()}\n")


@cocotb.test()
async def test_784_96_10_mnist(dut):
    await test_mnist(dut, 0.94, "../../nets/mnist_784_96_10_acc=94.57.qsd.pkl", 0, accepted_layers=['quant_net.0', 'quant_net.2'], skip_if_padded=True)


@cocotb.test()
async def test_70k_tcn_mnist(dut):
    await test_mnist(dut, 0.98, "../../nets/mnist_tcn_acc=98.75.qsd.pkl", 1, require_single_chunk=True)


@cocotb.test()
async def test_67k_tcn_mnist(dut):
    await test_mnist(dut, 0.98, "../../nets/mnist_tcn_785_acc=99.29.qsd.pkl", 1, require_single_chunk=True)


@cocotb.test()
async def test_14k4_tcn_mnist(dut):
    await test_mnist(dut, 0.98, "../../nets/mnist_14k4_tcn_98.77.qsd.pkl", 1, require_single_chunk=True, in_subsection_mode=True)
