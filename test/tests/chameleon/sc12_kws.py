import pickle

import cocotb

from .utils import get_params_and_controller, caller_name


CLOCK_FREQ = 100*10**6
SLOG2_WEIGHTS = True
CFG_MEMORY_FILE = "../../src/config_memory.json"
POINTER_FILE = "../../src/pointers.json"


async def test_sc12_kws(
    dut,
    quant_state_dict_file_path: str,
    n_samples_to_test: int,
    expected_accuracy: float,
    in_subsection_mode: bool = False,
    require_single_chunk: bool = True,
    raw_audio: bool = False
):
    _, c = get_params_and_controller(dut, CFG_MEMORY_FILE, POINTER_FILE)

    await c.start_and_verify_asic(1, CLOCK_FREQ, reset_host=False, verify_memories=False)

    if raw_audio:
        from AudioLoader.speech import SPEECHCOMMANDS_12C

        TAR_NAME = 'speech_commands_v0.02'
        FOLDER_NAME = 'SpeechCommands'

        TUPLE2LABEL = lambda _0, label, _2, _3, _4: label

        ds = SPEECHCOMMANDS_12C(
            root="../../data/",
            url=TAR_NAME,
            subset='validation',
            folder_in_archive=FOLDER_NAME,
            download=True,
            target_transform=TUPLE2LABEL
        )

        # Create a new dataset taking 34 samples from each class, so first extract the labels per sample

        indices_per_class = {}

        for i, (data, label) in enumerate(ds):
            if label not in indices_per_class:
                indices_per_class[label] = []
            indices_per_class[label].append(i)

        # Now take 34 samples from each class

        dataset_new = []

        for label, indices in indices_per_class.items():
            for i in indices[:34]:
                dataset_new.append(ds[i])

        dataset = dataset_new
    else:
        with open("../../chameleon/datasets/sc12_test_samples.pkl", "rb") as f:
            dataset = pickle.load(f)

    prev_accuracy = -1

    write_state_dict_to_asic = True

    for fill_input_memory in (False, True):
        for power_down_memories_while_running in (False, True):
            cocotb.log.info(f"> Running with: \n - fill_input_memory: {fill_input_memory}\n - power_down_memories_while_running: {power_down_memories_while_running}")

            accuracy, _ = await c.classify_dataset(
                dataset,
                quant_state_dict_file_path,
                n_samples_to_test=n_samples_to_test,
                write_state_dict_to_asic=write_state_dict_to_asic,
                configure_processing_setup=True,
                fill_input_memory=fill_input_memory,
                power_down_memories_while_running=power_down_memories_while_running,
                require_single_chunk=require_single_chunk,
                in_subsection_mode=in_subsection_mode,
                clip_inputs=True,
                input_padding_strategy='post',
                padding_value=0,
                expected_min_accuracy=expected_accuracy
            )

            write_state_dict_to_asic = False

            if prev_accuracy != -1:
                assert accuracy == prev_accuracy, f"Accuracy is not consistent: {accuracy} != {prev_accuracy}"

            prev_accuracy = accuracy

    with open("passed.txt", "a") as f:
        f.write(f"{caller_name()}\n")


@cocotb.test()
async def test_15k8_sc12_kws_subsection(dut):
    await test_sc12_kws(
        dut,
        "../../nets/12c_kws_mfcc_0usnfejd_step=5512_acc=93.31.qsd.pkl",
        n_samples_to_test=15,
        expected_accuracy=0.9332,
        in_subsection_mode=True
    )

@cocotb.test()
async def test_15k8_sc12_kws(dut):
    await test_sc12_kws(
        dut,
        "../../nets/12c_kws_mfcc_0usnfejd_step=5512_acc=93.31.qsd.pkl",
        n_samples_to_test=15,
        expected_accuracy=0.9332,
        in_subsection_mode=False,
        require_single_chunk=False
    )


@cocotb.test()
async def test_raw_audio_sc12_kws(dut):
    await test_sc12_kws(
        dut,
        "../../nets/12c_kws_raw_audio_4swylpi6_step=6312_acc=84.00.qsd.pkl",
        n_samples_to_test=9,
        expected_accuracy=0.8887,
        in_subsection_mode=False,
        raw_audio=True
    )
