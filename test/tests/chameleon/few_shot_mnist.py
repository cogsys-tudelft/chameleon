import cocotb

from torchvision import datasets, transforms
import torch.nn as nn

from torch_mate.data.utils import FewShot

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
        FewShot(mnist_test, 5, 1, 5), 5, 1, 5, 1, 0, CLOCK_FREQ,
        ((0.999, 0.999), (0.999, 0.999)), "../../nets/mnist_14k4_tcn_98.77.qsd.pkl",
        n_last_layers_to_remove=1, require_single_chunk=True, seed=0, l2_options='both',
        check_memory_contents=True, in_subsection_mode=True
    )


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_icl_4_test_shots(dut):
    ways = 5
    shots = 1
    query_shots = 4

    few_shot_data = FewShot(mnist_test, ways, shots, query_shots)

    await learn_with_few_shots(
        dut=dut,
        cfg_memory_file=CFG_MEMORY_FILE,
        pointer_file=POINTER_FILE,
        few_shot_dataset=few_shot_data,
        shots=shots,
        query_shots=query_shots,
        ways=ways,
        num_batches=2,
        ways_for_continued_learning=0,
        clock_freq=CLOCK_FREQ,
        expected_accuracies=(None, None),
        quant_state_dict_file_path= "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        n_last_layers_to_remove=1,
        require_single_chunk=True,
        seed=2,
        l2_options=False,
        check_memory_contents=True,
        quant_icl_layers=dict(
            conv_blocks=[],
            conv_kernel_sizes=[],
            linear_blocks=[2, 3, 1],
        ),
        icl_classification_options='both')


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_zero_shot_4_test_shots(dut):
    ways = 4
    shots = 1
    query_shots = 5

    few_shot_data = FewShot(mnist_test, ways, shots, query_shots)

    await learn_with_few_shots(
        dut=dut,
        cfg_memory_file=CFG_MEMORY_FILE,
        pointer_file=POINTER_FILE,
        few_shot_dataset=few_shot_data,
        shots=shots,
        query_shots=query_shots,
        ways=ways,
        num_batches=2,
        ways_for_continued_learning=0,
        clock_freq=CLOCK_FREQ,
        expected_accuracies=(None, None),
        quant_state_dict_file_path="../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        n_last_layers_to_remove=1,
        n_last_layers_to_remove_query=0,
        require_single_chunk=True,
        seed=2,
        l2_options=False,
        check_memory_contents=True,
        quant_icl_layers=dict(
            conv_blocks=[],
            conv_kernel_sizes=[],
            linear_blocks=[2, 3, 1],
        ),
        icl_classification_options='both',
        query_sample_embedder_quant_state_dict_file_path="../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl"
        )


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_icl_4_test_shots(dut):
    ways = 5
    shots = 1
    query_shots = 4

    few_shot_data = FewShot(mnist_test, ways, shots, query_shots)

    await learn_with_few_shots(
        dut=dut,
        cfg_memory_file=CFG_MEMORY_FILE,
        pointer_file=POINTER_FILE,
        few_shot_dataset=few_shot_data,
        shots=shots,
        query_shots=query_shots,
        ways=ways,
        num_batches=2,
        ways_for_continued_learning=0,
        clock_freq=CLOCK_FREQ,
        expected_accuracies=(None, None),
        quant_state_dict_file_path= "../../nets/mnist_tcn_acc=98.75.qsd.pkl",
        n_last_layers_to_remove=1,
        require_single_chunk=True,
        seed=2,
        l2_options=False,
        check_memory_contents=True,
        quant_icl_layers=dict(
            conv_blocks=[],
            conv_kernel_sizes=[],
            linear_blocks=[2, 3, 1],
        ),
        icl_classification_options='both')


@cocotb.test()
async def test_70k_tcn_1_way_4_shot_mnist_zero_shot_4_test_shots_reversed_nets(dut):
    ways = 4
    shots = 1
    query_shots = 5

    few_shot_data = FewShot(mnist_test, ways, shots, query_shots)

    await learn_with_few_shots(
        dut=dut,
        cfg_memory_file=CFG_MEMORY_FILE,
        pointer_file=POINTER_FILE,
        few_shot_dataset=few_shot_data,
        shots=shots,
        query_shots=query_shots,
        ways=ways,
        num_batches=2,
        ways_for_continued_learning=0,
        clock_freq=CLOCK_FREQ,
        expected_accuracies=(None, None),
        quant_state_dict_file_path="../../nets/omniglot_tcn_20_way_1_shot_acc=93.51.qsd.pkl",
        n_last_layers_to_remove=0,
        n_last_layers_to_remove_query=1,
        require_single_chunk=True,
        seed=2,
        l2_options=False,
        check_memory_contents=True,
        quant_icl_layers=dict(
            conv_blocks=[],
            conv_kernel_sizes=[],
            linear_blocks=[2, 3, 1],
        ),
        icl_classification_options='both',
        query_sample_embedder_quant_state_dict_file_path="../../nets/mnist_tcn_acc=98.75.qsd.pkl"
        )
