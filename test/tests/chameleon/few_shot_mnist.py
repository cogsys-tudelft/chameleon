import cocotb

from torchvision import datasets, transforms
import torch.nn as nn

from .utils import learn_with_few_shots


CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"

# Normalize images and flatten them
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    nn.Flatten(start_dim=1)
])

mnist_test = datasets.MNIST('../../data', train=False, download=True, transform=mnist_transform)


@cocotb.test()
async def test_70k_tcn_5_way_1_shot_mnist(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 1, 2, 5, 1, 0, CLOCK_FREQ,
        ((0.899, 0.899), (0.899, 0.899)), "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        1, True, 0, 'both', True
    )


@cocotb.test()
async def test_70k_tcn_5_way_5_shot_mnist(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 5, 1, 5, 1, 0, CLOCK_FREQ,
        ((0.999, 0.999), (0.999, 0.999)), "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        n_last_layers_to_remove=1, require_single_chunk=True, seed=0, l2_options='both',
        check_memory_contents=True
    )


@cocotb.test()
async def test_70k_tcn_2_way_10_shot_mnist(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 10, 1, 2, 1, 0, CLOCK_FREQ,
        ((0.999, 0.999), (0.999, 0.999)), "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        n_last_layers_to_remove=1, require_single_chunk=True, seed=0, l2_options='both',
        check_memory_contents=True
    )


@cocotb.test()
async def test_14k4_tcn_5_way_5_shot_mnist(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 5, 1, 5, 1, 0, CLOCK_FREQ,
        ((0.999, 0.999), (0.999, 0.999)), "../../nets/mnist_14k4_tcn_98.77.qsd.pkl",
        n_last_layers_to_remove=1, require_single_chunk=True, seed=0, l2_options='both',
        check_memory_contents=True, in_subsection_mode=True
    )


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_relation_net_4_test_shots(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 8, 1, 1, 2, 0, CLOCK_FREQ,
        (None, None), "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        1, True, 0, False, True, icl_shots=4
    )


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_icl_4_test_shots(dut):
    await learn_with_few_shots(
        dut,
        CFG_MEMORY_FILE, POINTER_FILE,
        mnist_test, 8, 1, 1, 1, 0, CLOCK_FREQ,
        (None, None), "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        1, True, 0, False, True, icl_shots=4
    )
