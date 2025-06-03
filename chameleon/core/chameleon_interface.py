import json
from pathlib import Path
from typing import Optional, List, Union, Dict, OrderedDict, Tuple, Literal, Callable
from dataclasses import dataclass
import math

import numpy as np
import numpy.testing as npt

from torch.utils.data import Dataset

from tqdm import tqdm

from torch_mate.data.utils import FewShot

from asic_cells.spi import SpiMessageCreator
from asic_cells.utils import chunk_list, to_binary_string

from chameleon.core.quant_conversions import int_to_slog2

from chameleon.core.net_transfer_utils import get_network_config, weight_rows_to_messages, bias_rows_to_messages, get_weight_rows, get_output_data, get_input_messages, get_few_shot_scales, get_quant_state_dict_and_layers, activations_to_messages
from chameleon.core.net_load_utils import get_random_tcn, get_random_input_tensor, get_output_size, get_random_input_tensor_by_length, get_quant_input
from chameleon.core.shared_utils import twos_complement_to_int, QuantLayers, iclog2, clog2, correct_subsection_mode_argmax, set_seed, assert_asic_out
from chameleon.core.learning_utils import left_right_shift, compute_expected_weight_and_bias, get_subsection_blocks_2d, get_subsection_blocks_1d
from chameleon.core.numpy.tcn import tcn_network, fc_bias_relu


@dataclass
class ChameleonParams:
    MESSAGE_BIT_WIDTH: int
    CODE_BIT_WIDTH: int
    START_ADDRESS_BIT_WIDTH: int
    ACTIVATION_BIT_WIDTH: int
    SCALE_BIT_WIDTH: int
    BIAS_BIT_WIDTH: int
    PE_COLS: int
    PE_ROWS: int
    SUBSECTION_SIZE: int
    WEIGHT_BIT_WIDTH: int
    HIGH_SPEED_IN_PINS: int
    HIGH_SPEED_OUT_PINS: int
    ACTIVATION_ROWS: int
    WEIGHT_ROWS: int
    BIAS_ROWS: int
    INPUT_ROWS: int
    ACCUMULATION_BIT_WIDTH: int
    MAX_KERNEL_SIZE: int
    MAX_NUM_CHANNELS: int
    FEW_SHOT_ACCUMULATION_BIT_WIDTH: int
    MAX_SHOTS: int
    MAX_NUM_LOGITS: int
    WAIT_CYCLES_WIDTH: int


class ChameleonInterface:
    def __init__(self, params: ChameleonParams, memory_map: dict, config_memory_file: Union[str, Path], pointer_file: Union[str, Path], verbose: bool = False):
        self.params = params
        self.memory_map = memory_map

        with open(config_memory_file, 'r') as f:
            config_map = json.load(f)['config_sizes_and_names']

        with open(pointer_file, 'r') as f:
            pointer_map = json.load(f)['pointer_sizes_and_names']

        self.verbose = verbose

        self.spi_message_creator = SpiMessageCreator(self.params.MESSAGE_BIT_WIDTH,
                                                     self.params.CODE_BIT_WIDTH,
                                                     self.params.START_ADDRESS_BIT_WIDTH,
                                                     config_map,
                                                     pointer_map,
                                                     memory_map)

        self.subsection_enabled = False
        
    def log_info(self, msg: object):
        raise NotImplementedError("Log info function not implemented for Chameleon Interface")

    async def sleep(self, duration: float):
        raise NotImplementedError("Sleep function not implemented for Chameleon Interface")
    
    def set_asic_inputs(self, **kwargs):
        raise NotImplementedError("ASIC inputs not implemented for Chameleon Interface")
    
    def get_asic_outputs(self, *args: str) -> dict:
        raise NotImplementedError("ASIC outputs not implemented for Chameleon Interface")
    
    def enable_supply(self, core: Optional[bool] = None, macro: Optional[bool] = None, io: Optional[bool] = None):
        raise NotImplementedError("Toggling supply not implemented for Chameleon Interface")
    
    # SEPARATE CLASS SHARED
    async def transfer_spi_data(self, messages: List[str]) -> List[int]:
        raise NotImplementedError("SPI data transfer not implemented for Chameleon Interface")
    
    async def select_external_clock(self, clock: int, cycles: Optional[int] = None, timeout: float = 0.1):
        raise NotImplementedError("Selecting external clock not implemented for Chameleon Interface")
    
    def zero_inputs(self):
        raise NotImplementedError("Zeroing inputs not implemented for Chameleon Interface")
    
    async def reset_host(self, timeout: float = 0.1):
        raise NotImplementedError("Host reset not implemented for Chameleon Interface")
    
    async def send_high_speed_inputs(self, input_messages: List[int]):
        raise NotImplementedError("High-speed input sending not implemented for Chameleon Interface")
    
    async def receive_high_speed_outputs(self, expected_num_outputs: int) -> List[int]:
        raise NotImplementedError("High-speed output receiving not implemented for Chameleon Interface")
    
    def configure_voltage_setpoints(self, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3):
        raise NotImplementedError("Voltage setpoints not implemented for Chameleon Interface")
    
    async def reset_asic(self, timeout: float = 0.1):
        """Reset the ASIC by toggling the async reset signal.

        Args:
            timeout (float, optional): The time to wait after the reset. Defaults to 0.1.
        """

        self.set_asic_inputs(rst_async=1)
        await self.sleep(timeout)
        self.set_asic_inputs(rst_async=0)
    
    async def transfer_spi_data_if_available(self, messages: List[str]):
        assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state and so cannot receive SPI data"

        return await self.transfer_spi_data(messages)
    
    # SEPARATE CLASS SHARED
    async def send_asic_config_over_spi(self, config: Dict[str, Union[int, List[int]]]):
        await self.transfer_spi_data_if_available(self.spi_message_creator.create_config_messages(config))

    # SEPARATE CLASS SHARED
    async def get_asic_pointers_over_spi(self, pointers: List[str]) -> Dict[str, int]:
        responses = []

        for pointer in pointers:
            messages = [self.spi_message_creator.create_pointer_message(pointer), "0" * self.params.MESSAGE_BIT_WIDTH]
            responses.extend(await self.transfer_spi_data_if_available(messages))

        return dict(zip(pointers, responses))
    
    def turn_off_supplies(self):
        self.enable_supply(core=False, macro=False, io=False)
    
    async def power_cycle_asic(self, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3, timeout: float = 0.01):
        self.turn_off_supplies()
        await self.sleep(timeout)
        self.configure_voltage_setpoints(core_voltage, macro_voltage, io_voltage)

        self.enable_supply(core=True, macro=True, io=True)

    async def reset_and_select_external_clock(self, clock: int, cycles: Optional[int] = None, timeout: float = 0.1):
        await self.reset_asic(timeout)
        await self.select_external_clock(clock, cycles, timeout)
        await self.reset_asic(timeout)

    async def start_up_asic(self, clock: int, cycles: Optional[int] = None, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3, timeout: float = 0.1):
        """Start up the ASIC.

        Args:
            clock (int): Which external (going from host to ASIC) clock to use.
            cycles (Optional[int], optional): The number of cycles for half a cycle of custom clock. Defaults to None.
            timeout (float, optional): Time to wait between between the different start-up steps. Defaults to 0.1.
        """

        self.zero_inputs()

        await self.power_cycle_asic(core_voltage, macro_voltage, io_voltage)
        await self.reset_and_select_external_clock(clock, cycles, timeout)
        assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state"


    async def verify_spi(self):
        """Check based on some known pointer values whether the SPI communication is working correctly.
        """

        # Check whether the constant pointer 'douwe' is correct
        key = "douwe"

        assert (await self.get_asic_pointers_over_spi([key]))[key] == int('10011101100001111010000000111000', 2), f"SPI pointer '{key}' not correct"

        await self.send_asic_config_over_spi({"max_bias_address": self.memory_map["biases"]["num_rows"]-1})

        # In the Verilog code of the chip, the pointer 'configured_for_few_shot_processing' is defined as follows:
        # assign configured_for_few_shot_processing = classification && (shots != 0);
        # So when we set classification to True and shots to 2, the pointer should be True
        # We do not set "shots" to 1, as that would cause start_few_shot_processing to go high
        # and possibly cause the chip to enter few-shot processing mode
        await self.send_asic_config_over_spi({"classification": True, "shots": 2})
        
        key = "configured_for_few_shot_processing"

        assert (await self.get_asic_pointers_over_spi([key]))[key], "ASIC configuration not correct after enabling few-shot processing"

        # Reset the pointer
        await self.send_asic_config_over_spi({"classification": False, "shots": 0, "max_bias_address": 0})

        assert not (await self.get_asic_pointers_over_spi([key]))[key], "ASIC configuration not correct after disabling few-shot processing"

    # SEPARATE CLASS SHARED
    async def write_asic_memory_over_spi(self, key: str, data: List[int], start_address: int = 0, check_back: bool = False):
        await self.transfer_spi_data_if_available(self.spi_message_creator.create_write_memory_messages(key, data, start_address))

        if check_back:
            assert await self.read_asic_memory_over_spi(key, start_address, len(data)) == data, f"Read-back data from memory '{key}' is not the same as written data"

    # SEPARATE CLASS SHARED
    async def read_asic_memory_over_spi(self, key: str, start_address: int, num_transactions: int = 1) -> List[int]:
        spi_output = []

        max_transactions = 2**self.spi_message_creator.num_transactions_bit_width-1

        num_transactions_remaining = num_transactions

        for current_start_address in range(start_address, start_address + num_transactions, max_transactions):
            if num_transactions_remaining > max_transactions:
                current_num_transactions = max_transactions
            else:
                current_num_transactions = num_transactions_remaining

            num_transactions_remaining -= current_num_transactions

            message = self.spi_message_creator.create_read_memory_message(key, current_start_address, current_num_transactions)
            spi_output.extend(await self.transfer_spi_data_if_available([message] + ["0"*self.params.MESSAGE_BIT_WIDTH]*current_num_transactions))

        return spi_output

    # SEPARATE CLASS SHARED
    async def verify_asic_memories_over_spi(self, memories: Optional[list[str]] = None):
        """Verify the memories of the ASIC over SPI.

        Args:
            memories (Optional[list[str]], optional): Keys of memories to check. Defaults to None (which means all memories are checked).
        """

        if memories is None:
            memories = self.memory_map.keys()

        for memory in memories:
            num_messages = self.spi_message_creator._memory_indices_and_max_addresses[memory][1]
            random_data_ints = [int(x, 2) for x in [self.spi_message_creator.create_random_data_message() for _ in range(num_messages)]]

            await self.write_asic_memory_over_spi(memory, random_data_ints, 0, True)

    # SEPARATE CLASS SHARED
    async def set_asic_memories_to_zero_over_spi(self, memories: Optional[list[str]] = None, check_back: bool = False):
        """Sets specified ASIC memories to zero.

        Args:
            memories (Optional[list[str]], optional): Keys of memories to check. Defaults to None (which means all memories are checked).
            check_back (bool, optional): Whether to check if zeroes have been correctly written. Defaults to False.
        """

        if memories is None:
            memories = self.memory_map.keys()

        for memory in memories:
            # Use the pre-computed maximum number of messages for each memory
            num_messages = self.spi_message_creator._memory_indices_and_max_addresses[memory][1]

            await self.write_asic_memory_over_spi(memory, [0] * num_messages, 0, check_back)

    # SEPARATE CLASS SHARED
    async def read_asic_memory_as_numpy_array(self, key: str, start_row: int, length: int = 1, entry_bit_width: int = 8, signed: bool = False) -> np.ndarray:
        # TODO: clean up this code!
        last_row = start_row + length

        log2_bit_width = iclog2(self.memory_map[key]["bit_width"])
        messages_per_row = log2_bit_width // self.spi_message_creator.message_bit_width
        inputs_per_input_row = self.memory_map[key]["bit_width"] // entry_bit_width

        raw_messages = await self.read_asic_memory_over_spi(key, messages_per_row*start_row, messages_per_row*(last_row-start_row))
        rows = chunk_list([to_binary_string(message, self.spi_message_creator.message_bit_width) for message in raw_messages], messages_per_row)

        correct_within_row_order = list(map(lambda x: list(reversed(x)), rows))
        rows_as_strings = list(map(lambda x: ''.join(x), correct_within_row_order))

        if log2_bit_width != self.memory_map[key]["bit_width"]:
            bits_to_remove = log2_bit_width - self.memory_map[key]["bit_width"]
            rows_as_strings = [row[bits_to_remove:] for row in rows_as_strings]

        reversed_row_order = list(reversed(rows_as_strings))
        single_string = ''.join(reversed_row_order)
        as_ints = list(map(lambda x: int(x, 2), chunk_list(single_string, entry_bit_width)))

        if signed:
            as_ints = [twos_complement_to_int(x, entry_bit_width) for x in as_ints]

        correct_row_order = list(reversed(as_ints))
        as_nested_list = chunk_list(correct_row_order, inputs_per_input_row)

        return np.array(as_nested_list)

    async def start_and_verify_asic(self, clock: int, cycles: Optional[int] = None, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3, reset_host: bool = True, timeout: float = 0.1, verify_memories: bool = True):
        if reset_host:
            self.reset_host(timeout)
            self.log_info("> Reset host")

        await self.start_up_asic(clock, cycles, core_voltage, macro_voltage, io_voltage, timeout)
        self.log_info("> Started ASIC")

        await self.verify_spi()
        self.log_info("> Verified SPI")

        if verify_memories:
            await self.verify_asic_memories_over_spi()
            self.log_info("> Verified all memories")

            await self.set_asic_memories_to_zero_over_spi(check_back=True)
            self.log_info("> Zero'ed all memories")
        else:
            self.log_info("> Skipped memory verification")

    async def enable_subsection_mode(self, power_down_small_bias: bool = False, wake_up_delay_cycles: int = 10, power_up_delay_cycles: int = 10):
        """Enable the subsection mode of the ASIC.

        Args:
            power_down_small_bias (bool, optional): Whether to also power down the second small bias. Only possible when you have less than 64 bias rows. Defaults to False.
        """

        assert 0 <= wake_up_delay_cycles < 2**self.params.WAIT_CYCLES_WIDTH
        assert 0 <= power_up_delay_cycles < 2**self.params.WAIT_CYCLES_WIDTH

        sleep_duration = 0.001

        if self.subsection_enabled == False:
            self.log_info("> Enabling subsection mode")

            await self.send_asic_config_over_spi({"wake_up_delay": wake_up_delay_cycles, "power_up_delay": power_up_delay_cycles, "power_down_srams_in_standby": True, "in_4x4_mode": True, "require_single_chunk": True, "power_down_small_bias": power_down_small_bias})
            await self.sleep(sleep_duration)
            self.enable_supply(macro=False)
            await self.sleep(sleep_duration)
            await self.send_asic_config_over_spi({"power_down_srams_in_standby": False})
            await self.sleep(sleep_duration)

            self.subsection_enabled = True

    async def disable_subsection_mode(self):
        """Disable the subsection mode of the ASIC.
        """

        if self.subsection_enabled:
            self.log_info("> Disabling subsection mode")

            await self.send_asic_config_over_spi({"in_4x4_mode": False, "require_single_chunk": False, "power_down_small_bias": False})
            await self.sleep(0.001)
            self.enable_supply(macro=True)
            await self.sleep(0.001)

            self.subsection_enabled = False

    async def get_input_memory_contents(self, start_row: int, length: int = 1):
        return await self.read_asic_memory_as_numpy_array("inputs", start_row, length, entry_bit_width=self.params.ACTIVATION_BIT_WIDTH)

    async def get_activation_memory_contents(self, start_row: int, length: int = 1):
        return await self.read_asic_memory_as_numpy_array("activations", start_row, length, entry_bit_width=self.params.ACTIVATION_BIT_WIDTH)

    async def get_bias_memory_contents(self, start_row: int, length: int = 1):
        return await self.read_asic_memory_as_numpy_array("biases", start_row, length, entry_bit_width=self.params.BIAS_BIT_WIDTH, signed=True)

    async def get_weight_memory_contents(self, start_row: int, length: int = 1):
        weights = await self.read_asic_memory_as_numpy_array("weights", start_row, length, entry_bit_width=self.params.WEIGHT_BIT_WIDTH)

        # Apply corrections for the weight layout as required by subsection operation mode
        flattened_weights = np.reshape(weights, (len(weights), -1))
        sub_block = np.transpose(np.reshape(flattened_weights[:, :self.params.SUBSECTION_SIZE*self.params.PE_COLS], (-1, self.params.PE_COLS, self.params.SUBSECTION_SIZE)), (0, 2, 1))
        remaining = np.transpose(np.reshape(flattened_weights[:, self.params.SUBSECTION_SIZE*self.params.PE_COLS:], (-1, self.params.PE_COLS, self.params.PE_ROWS-self.params.SUBSECTION_SIZE)), (0, 2, 1))
        corrected_weights = np.concatenate((sub_block, remaining), axis=1)

        return corrected_weights

    async def forward(
        self,
        x: np.ndarray,
        expected_output_channels: int = -1,
        is_new_task: bool = False,
        in_subsection_mode: bool = False,
        expect_pre_post_idle_state: bool = True,
        require_single_chunk: bool = False,
        toggle_processing: bool = True,
        activation_memory_address: Optional[int] = None,
        timeout: float = 0.0002
    ) -> Union[np.ndarray, int]:
        """Perform a forward pass on the ASIC.

        Args:
            x (np.ndarray): Input data to be processed by the ASIC of shape (input_channels, sequence_length).
            expected_output_channels (int, optional): Expected number of output channels. Set to -1 for classification and to 0 if you are feeding data to learn a new task. Defaults to -1.
            is_new_task (Optional[bool], optional): Whether the forward pass is for a new task. Defaults to None.
            in_subsection_mode (bool, optional): Whether the forward pass takes place in subsection mode. Defaults to False.
            expect_pre_post_idle_state (bool, optional): Whether to check if the ASIC is in idle state before and after the forward pass. Defaults to True.
            require_single_chunk (bool, optional): Whether to require that the input data fits in a single chunk. Defaults to False.
            activation_memory_address (Optional[int], optional): Starting address in the activation memory to write the input data to. Defaults to None.
            timeout (float, optional): Time to wait between the different steps of the forward pass. Defaults to 0.01.

        Returns:
            Union[np.ndarray, int]: Output data of shape (output_channels,) or argmax of the output data if classification is True.
        """

        if len(x.shape) == 1:
            x = np.expand_dims(x, 1)

        input_channels, _ = x.shape

        if require_single_chunk:
            if not in_subsection_mode:
                assert input_channels <= self.params.HIGH_SPEED_IN_PINS // self.params.ACTIVATION_BIT_WIDTH
        elif input_channels % self.params.PE_ROWS != 0:
            new_channels = math.ceil(input_channels / self.params.PE_ROWS) * self.params.PE_ROWS
            x = np.pad(x, ((0, new_channels-input_channels), (0, 0)), mode='constant', constant_values=0).astype(x.dtype)

        assert expected_output_channels < 0 or expected_output_channels % (self.params.SUBSECTION_SIZE if in_subsection_mode else self.params.PE_ROWS) == 0, "Expected output channels must be a multiple of the PE size"
 
        if is_new_task != False:
            assert expected_output_channels == 0, "Expected output channels must be 0 when is_new_task is set"

        output_bit_width = self.params.ACTIVATION_BIT_WIDTH
        channels_per_output_block = self.params.HIGH_SPEED_OUT_PINS // output_bit_width

        in_channels_per_block = self.params.HIGH_SPEED_IN_PINS // self.params.ACTIVATION_BIT_WIDTH

        if activation_memory_address is not None:
            if activation_memory_address != -1:
                all_input_messages = activations_to_messages(x, activation_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                                            in_subsection_mode=in_subsection_mode,
                                                            pe_array_size=self.params.PE_ROWS,
                                                            subsection_size=self.params.SUBSECTION_SIZE,
                                                            spi_message_bit_width=self.spi_message_creator.message_bit_width)

                address_message_factor = self.memory_map["activations"]["bit_width"] // self.spi_message_creator.message_bit_width
                await self.write_asic_memory_over_spi("activations", all_input_messages, activation_memory_address * address_message_factor, check_back=False)

        if expect_pre_post_idle_state:
            assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state and so cannot process new inputs"

        if toggle_processing:
            self.set_asic_inputs(is_new_task=False, toggle_processing=False)
            await self.sleep(timeout)
            self.set_asic_inputs(is_new_task=is_new_task, toggle_processing=True)

        if is_new_task == True:
            # Make sure that is_new_task is low before the chip goes into idle after
            # processing these inputs
            await self.sleep(timeout)
            self.set_asic_inputs(is_new_task=False)

        if activation_memory_address is None:
            all_input_messages = get_input_messages(x, self.params.ACTIVATION_BIT_WIDTH, in_channels_per_block, padding_value=0)

            await self.send_high_speed_inputs(all_input_messages)

        if self.verbose:
            self.log_info("> Forward pass completed")

        # If a new task is being inputted, no outputs are expected
        # Zero expected_output_channels indicates sending a sample for few-shot learning
        if expected_output_channels == 0:
            if expect_pre_post_idle_state:
                while not self.get_asic_outputs("in_idle"):
                    await self.sleep(1e-5)

            return None

        # Minus one indication a classification task
        if expected_output_channels < 0:
            classification = True

            if expected_output_channels == -1:
                expected_output_channels = channels_per_output_block
            elif expected_output_channels == -2:
                expected_output_channels = math.ceil((clog2(self.params.MAX_NUM_LOGITS) + self.params.ACCUMULATION_BIT_WIDTH) / self.params.HIGH_SPEED_OUT_PINS) * channels_per_output_block
            else:
                raise ValueError("Invalid expected output channels for classification task: expected output channels can only be -1 or -2")
        else:
            classification = False

        expected_output_blocks = expected_output_channels // channels_per_output_block
        outputs = await self.receive_high_speed_outputs(expected_output_blocks)
        
        self.set_asic_inputs(toggle_processing=False)

        if expect_pre_post_idle_state:
            while not self.get_asic_outputs("in_idle"):
                await self.sleep(1e-5)

            assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state after processing inputs"

        out_channels_per_block = self.params.HIGH_SPEED_OUT_PINS // self.params.ACTIVATION_BIT_WIDTH
        output_channels = get_output_data(outputs, self.params.ACTIVATION_BIT_WIDTH, out_channels_per_block)

        if classification:
            output_string = ''.join([f"{x:0{self.params.ACTIVATION_BIT_WIDTH}b}" for x in reversed(output_channels)])
            argmax_value = None

            if len(output_string) > self.params.HIGH_SPEED_OUT_PINS:
                num_bits_for_argmax = clog2(self.params.MAX_NUM_LOGITS)
                output = int(output_string[-num_bits_for_argmax:], 2)
                argmax_value = twos_complement_to_int(int(output_string[:-num_bits_for_argmax], 2), self.params.ACCUMULATION_BIT_WIDTH)
            else:
                output = int(output_string, 2)

            if in_subsection_mode:
                output = correct_subsection_mode_argmax(output, self.params.PE_COLS, self.params.SUBSECTION_SIZE)
            
            if argmax_value is not None:
                return output, argmax_value

            return output
        
        return output_channels
    
    async def __call__(self, x: np.ndarray, **kwargs):
        return await self.forward(x, **kwargs)

    async def configure_processing_setup(self, classification: bool,
                                         continuous_processing: bool = False,
                                         fill_input_memory: bool = False,
                                         force_downsample: bool = False,
                                         power_down_memories_while_running: bool = False,
                                         in_subsection_mode: bool = False,
                                         continued_learning: bool = False,
                                         shots: int = 0,
                                         few_shot_scale: Optional[int] = None,
                                         power_down_small_bias: bool = False,
                                         require_single_chunk: bool = False,
                                         use_l2_for_few_shot: bool = False,
                                         k_shot_division_scale: Optional[int] = None,
                                         in_context_learning: bool = False,
                                         load_inputs_from_activation_memory: bool = False,
                                         send_all_argmax_chunks: bool = False):
        if (classification == False or continuous_processing == True) and shots > 0 and not in_context_learning:
            raise ValueError("Setting shots while classification is False or while continuous_processing is True has no effect!")
        elif continued_learning == True and (classification == False or continuous_processing == True or shots == 0):
            raise ValueError("Cannot have continued_learning == True and not be correctly configured for few-shot learning")
        
        if in_context_learning:
            assert shots != 0 and not continued_learning, "In-context learning requires few-shot learning to be enabled and continued learning to be disabled"

        if send_all_argmax_chunks and not classification:
            raise ValueError("Cannot send all argmax chunks when classification is disabled")
        
        max_cycles = self.params.WAIT_CYCLES_WIDTH**2 - 1

        if in_subsection_mode:
            # If input pins match or exceed subsection activation size
            if self.params.ACTIVATION_BIT_WIDTH * self.params.SUBSECTION_SIZE <= self.params.HIGH_SPEED_IN_PINS:
                assert require_single_chunk, "Require single chunk must be enabled when using subsection mode"

            wake_up_delay_cycles = max_cycles
            power_up_delay_cycles = max_cycles

            await self.enable_subsection_mode(power_down_small_bias, wake_up_delay_cycles, power_up_delay_cycles)
        else:
            await self.disable_subsection_mode()

            if power_down_small_bias:
                raise ValueError("Cannot power down small bias without enabling the subsection mode!")

        few_shot_scale, (few_shot_scale_sign_corrected, k_shot_division_scale) = get_few_shot_scales(shots, self.params.WEIGHT_BIT_WIDTH, self.params.ACTIVATION_BIT_WIDTH, self.params.MAX_SHOTS, few_shot_scale, k_shot_division_scale)

        processing_config = {
            "continuous_processing": continuous_processing,
            "classification": classification,
            "fill_input_memory": fill_input_memory,
            "force_downsample": force_downsample,
            "require_single_chunk": require_single_chunk,
            "power_down_memories_while_running": power_down_memories_while_running,
            "continued_learning": continued_learning,
            "shots": shots,
            "few_shot_scale": few_shot_scale_sign_corrected,
            "k_shot_division_scale": k_shot_division_scale,
            "use_l2_for_few_shot": use_l2_for_few_shot,
            "in_context_learning": in_context_learning,
            "load_inputs_from_activation_memory": load_inputs_from_activation_memory,
            "send_all_argmax_chunks": send_all_argmax_chunks
        }

        if power_down_memories_while_running:
            processing_config["wake_up_delay"] = max_cycles
            processing_config["power_up_delay"] = max_cycles

        await self.send_asic_config_over_spi(processing_config)

        self.log_info("> Configured processing setup")

        # In case shots == 0, few_shot_scale is None as no few-shot learning is being done
        if few_shot_scale == None:
            return None

        return few_shot_scale, -few_shot_scale-k_shot_division_scale

    async def reset_learned_ways(self, timeout: float = 0.01):
        assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state and so cannot reset learned ways"

        self.set_asic_inputs(is_new_task=True)
        await self.sleep(timeout)
        self.set_asic_inputs(is_new_task=False)

    async def write_network_to_asic(self, quant_layers: QuantLayers, padding_value: Optional[int] = None, subsection_network: bool = False, icl_layers_shots: Optional[Tuple[int, int]] = None, are_icl_shots_labeled: bool = False, activation_memory_address: Optional[int] = None, continued_learning: Optional[bool] = None):
        blocks = []
        biases = []
        scales = []
        downsample_scales = []

        num_input_blocks = -1
        num_conv_blocks_per_layer = []
        conv_kernel_sizes_per_layer = []
        linear_blocks = []
    
        if subsection_network:
            rows = self.params.SUBSECTION_SIZE
        else:
            rows = self.params.PE_ROWS

        # Chunk in groups of two assuming that every conv layer has actually two conv operations
        for i, layer in enumerate(quant_layers):
            if len(layer) == 3:
                weight, bias, scale = layer

                if num_input_blocks == -1:
                    num_input_blocks = weight.shape[1] / rows

                linear_blocks.append(weight.shape[0] / rows)

                blocks.extend(get_weight_rows(weight, self.params.PE_COLS, self.params.SUBSECTION_SIZE, padding_value, subsection_network))
                biases.append(bias)
                scales.append(scale)
                downsample_scales.append(0)
            elif len(layer) == 2:
                ((weight1, bias1, scale1), _), ((weight2, bias2, scale2), (downsample_weight, downsample_scale)) = layer

                if num_input_blocks == -1:
                    num_input_blocks = math.ceil(weight1.shape[1] / rows)

                num_conv_blocks_per_layer.extend([weight1.shape[0] / rows, weight2.shape[0] / rows])
                conv_kernel_sizes_per_layer.extend([weight1.shape[2], weight2.shape[2]])

                biases.extend([bias1, bias2])
                scales.extend([scale1, scale2])
                downsample_scales.extend([0, downsample_scale])

                intermediate_channels1, in_channels, kernel_size1 = weight1.shape
                out_channels, intermediate_channels2, kernel_size = weight2.shape

                assert kernel_size1 == kernel_size, f"Different kernel size in the same TCN layer ({i}) is not yet supported"
                assert intermediate_channels1 == intermediate_channels2, f"Intermediate channels in conv block must be the same for layer {i}"
                assert num_input_blocks * kernel_size <= self.params.INPUT_ROWS, f"Number of input blocks times kernel size must be less than or equal to the input rows for layer"

                downsample_weight_entries = None
                has_downsample = downsample_weight is not None

                if in_channels != out_channels:
                    assert has_downsample, f"Downsample weight must be provided for conv block with different input and output channels for layer {i}"

                if has_downsample:
                    downsample_out_channels, downsample_in_channels, _ = downsample_weight.shape

                    downsample_channel_error_message = lambda io: f"Downsample weight must have the same number of {io} channels as the conv block for layer {i}"

                    assert downsample_in_channels == in_channels, downsample_channel_error_message("input")
                    assert downsample_out_channels == out_channels, downsample_channel_error_message("output")

                    downsample_weight_entries = chunk_list(get_weight_rows(downsample_weight, self.params.PE_COLS, self.params.SUBSECTION_SIZE, padding_value, subsection_network), math.ceil(downsample_in_channels / rows))

                conv0_weight_rows = get_weight_rows(weight1, self.params.PE_COLS, self.params.SUBSECTION_SIZE, padding_value, subsection_network)
                conv1_weight_rows = get_weight_rows(weight2, self.params.PE_COLS, self.params.SUBSECTION_SIZE, padding_value, subsection_network)

                # First add the first conv operation layer weights
                blocks.extend(conv0_weight_rows)

                if downsample_weight_entries is not None:
                    splits = chunk_list(conv1_weight_rows, kernel_size * (intermediate_channels2 // rows))

                    for downsample_weight_row, conv1_weight_row in zip(downsample_weight_entries, splits):
                        blocks.extend(downsample_weight_row + conv1_weight_row)
                else:
                    blocks.extend(conv1_weight_rows)
            else:
                raise ValueError(f"Invalid layer configuration for layer {i}")
            
        # Divide by 2 due to WEIGHT_ROWS being twice as large as the actual rows
        # while in subsection mode
        half_way_rows = self.params.WEIGHT_ROWS // 2

        max_weight_address = len(blocks)

        # Make sure that when in subsection mode and more than half of the rows in the weight memory
        # are used, that they are still written in the correct format
        if subsection_network and len(blocks) >= half_way_rows:
            blocks = np.array(blocks)

            num_params_per_address = self.params.SUBSECTION_SIZE**2
            only_subsection_params = blocks[:, :num_params_per_address]

            first_half_blocks = only_subsection_params[:half_way_rows]    
            second_half_blocks = only_subsection_params[half_way_rows:]

            # Pad with zeros until length is half_way_rows for second_half_blocks
            second_half_blocks = np.concatenate((second_half_blocks, np.zeros((half_way_rows-len(second_half_blocks), num_params_per_address))), axis=0)

            blocks = np.concatenate((first_half_blocks, second_half_blocks), axis=1)
            blocks = np.concatenate((blocks, np.zeros((len(blocks), self.params.PE_ROWS**2 - 2*num_params_per_address))), axis=1)
            blocks = blocks.astype(int)

        all_weight_messages = weight_rows_to_messages(blocks, self.params.WEIGHT_BIT_WIDTH, self.spi_message_creator.message_bit_width)
        all_bias_messages = bias_rows_to_messages(biases, self.params.PE_COLS, self.params.SUBSECTION_SIZE, self.params.BIAS_BIT_WIDTH, self.spi_message_creator.message_bit_width, padding_value=-2**(self.params.BIAS_BIT_WIDTH-1), subsection_bias=subsection_network)

        network_config = get_network_config(
            [math.ceil(num_blocks) for num_blocks in num_conv_blocks_per_layer],
            conv_kernel_sizes_per_layer,
            [math.ceil(num_blocks) for num_blocks in linear_blocks],
            num_input_blocks,
            self.params.MAX_KERNEL_SIZE,
            math.ceil(self.params.MAX_NUM_CHANNELS / rows),
            self.params.ACTIVATION_ROWS,
            icl_layers_shots=icl_layers_shots,
            are_icl_shots_labeled=are_icl_shots_labeled,
            activation_memory_address=activation_memory_address,
            continued_learning=continued_learning
        )

        scale_and_residual_scale_per_layer = [(downsample_scale << self.params.SCALE_BIT_WIDTH) + scale if scale != -1 else 0 for (downsample_scale, scale) in zip(downsample_scales, scales)]
        network_config["scale_and_residual_scale_per_layer"] = scale_and_residual_scale_per_layer

        # Although sending this data is only needed for few-shot and continual learning,
        # we can send it anyway as it doesn't change anything in the processing setup
        bias_messages_per_row = iclog2(self.memory_map["biases"]["bit_width"]) // self.spi_message_creator.message_bit_width

        network_config["max_weight_address"] = max_weight_address
        network_config["max_bias_address"] = len(all_bias_messages) // bias_messages_per_row

        await self.send_asic_config_over_spi(network_config)

        self.log_info(f"> (1/3) Written network config to ASIC")

        await self.write_asic_memory_over_spi("biases", all_bias_messages, 0, check_back=False)

        self.log_info(f"> (2/3) Written biases to ASIC")

        await self.write_asic_memory_over_spi("weights", all_weight_messages, 0, check_back=False)

        self.log_info(f"> (3/3) Written weights to ASIC")

        return network_config

    async def write_quant_state_dict_to_asic(self, net_path_or_state_dict: Union[str, Path, OrderedDict],
                                             slog2_weights: bool = True, accepted_layers: Optional[List[str]] = None,
                                             n_last_layers_to_remove: Optional[int] = None, padding_value: Optional[int] = None,
                                             subsection_network: bool = False, continued_learning: Optional[bool] = None,
                                             activation_memory_address: Optional[int] = None):
        quant_in, quant_layers = get_quant_state_dict_and_layers(net_path_or_state_dict, slog2_weights,
                                                                 scale_bit_width=self.params.SCALE_BIT_WIDTH,
                                                                 accepted_layers=accepted_layers,
                                                                 n_last_layers_to_remove=n_last_layers_to_remove)

        network_config = await self.write_network_to_asic(
            quant_layers,
            padding_value=padding_value,
            subsection_network=subsection_network,
            continued_learning=continued_learning,
            activation_memory_address=activation_memory_address
        )

        return quant_in, quant_layers, network_config
    
    async def stop_continuous_processing(self, timeout: float = 0.001):
        self.set_asic_inputs(toggle_processing=True)
        await self.sleep(timeout)
        self.set_asic_inputs(toggle_processing=False)

        assert self.get_asic_outputs("in_idle")

    async def verify_net_structure(self,
                                input_blocks: int,
                                conv_blocks: list[int],
                                conv_kernel_sizes: list[int],
                                linear_blocks: list[int],
                                num_structure_repeats: int = 1,
                                num_forward_passes: int = 1,
                                is_new_task: bool = False,
                                shots: int = 0,
                                slog2_weights: bool = True,
                                seed: int = 0,
                                in_subsection_mode: Optional[bool] = None,
                                force_downsample: Optional[bool] = None,
                                check_memory_contents: bool = False,
                                num_continuous_forward_passes: int = 2,
                                classification_options: Union[str, bool] = 'both',
                                activation_memory_address: Optional[int] = None):
        """Verify a certain network structure under different processing configurations."""

        set_seed(seed)

        if force_downsample == None and len(conv_blocks) > 0:
            force_downsamples = [False, True] 
        elif force_downsample != None:
            force_downsamples = [force_downsample]
        else:
            force_downsamples = [False]

        if in_subsection_mode == None:
            in_subsection_modes = [False, True]
        else:
            in_subsection_modes = [in_subsection_mode]

        if num_continuous_forward_passes > 1:
            continuous_processings = [False, True]
        else:
            continuous_processings = [False]

            assert num_continuous_forward_passes == 0, "Number of continuous forward passes must be 0"

        if activation_memory_address is not None:
            activation_memory_addresses = [activation_memory_address]
        else:
            activation_memory_addresses = [None]

        if classification_options == 'both':
            current_classification_options = (False, True)
        else:
            current_classification_options = (classification_options,)

        # - largest possible networks
        for _ in range(num_structure_repeats):
            self.log_info(f"> Starting structure repeat {_+1}/{num_structure_repeats}")
        
            for in_subsection_mode in in_subsection_modes:
                for force_downsample in force_downsamples:
                    for activation_memory_address in activation_memory_addresses:
                        quant_layers = get_random_tcn(
                            input_blocks,
                            conv_blocks,
                            conv_kernel_sizes,
                            linear_blocks,
                            pe_rows=self.params.PE_ROWS,
                            weight_bit_width=self.params.WEIGHT_BIT_WIDTH,
                            act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                            bias_bit_width=self.params.BIAS_BIT_WIDTH,
                            subsection_size=self.params.SUBSECTION_SIZE if in_subsection_mode else -1,
                            force_downsample=force_downsample,
                            slog2_weights=slog2_weights
                        )

                        await self.write_network_to_asic(quant_layers, subsection_network=in_subsection_mode, activation_memory_address=activation_memory_address)

                        for classification in current_classification_options:
                            send_all_argmax_chunks = get_output_size(quant_layers) * (self.params.PE_ROWS / self.params.SUBSECTION_SIZE if in_subsection_mode else 1) > 2**self.params.HIGH_SPEED_OUT_PINS and classification

                            for fill_input_memory in (False, True):
                                if activation_memory_address is not None and fill_input_memory:
                                    self.log_info("> Skipping fill input memory test as activation memory is used")
                                    continue

                                for continuous_processing in continuous_processings:
                                    if continuous_processing:
                                        # If there are no convolutional layers, there is no need to run continuous processing tests
                                        if len(conv_kernel_sizes) == 0:
                                            self.log_info("> Skipping continuous processing test as there are no convolutional layers")
                                            continue

                                        if activation_memory_address is not None:
                                            self.log_info("> Skipping continuous processing test as activation memory is used")
                                            continue

                                    for power_down_memories_while_running in (False, True):
                                        self.log_info(f"> Running with: \n - classification: {classification}\n - activation_memory_address: {activation_memory_address}\n - fill_input_memory: {fill_input_memory}\n - force_downsample: {force_downsample}\n - shots: {shots}\n - is_new_task: {is_new_task}\n - in_subsection_mode: {in_subsection_mode}\n - continuous_processing: {continuous_processing}\n - power_down_memories_while_running: {power_down_memories_while_running}")

                                        if is_new_task and (shots == 0 or not classification):
                                            raise ValueError("Cannot indicate a new task while the number of shots is 0 or classification is False")

                                        # TODO: move all config processing variables into for loops
                                        await self.configure_processing_setup(
                                            classification=classification,
                                            continuous_processing=continuous_processing,
                                            fill_input_memory=fill_input_memory,
                                            power_down_memories_while_running=power_down_memories_while_running,
                                            in_subsection_mode=in_subsection_mode,
                                            continued_learning=False,
                                            force_downsample=force_downsample,
                                            shots=shots,
                                            power_down_small_bias=in_subsection_mode,
                                            require_single_chunk=in_subsection_mode,
                                            load_inputs_from_activation_memory=activation_memory_address is not None,
                                            send_all_argmax_chunks=send_all_argmax_chunks
                                        )

                                        for _ in range(num_forward_passes):
                                            input_tensor = get_random_input_tensor(
                                                input_blocks,
                                                conv_blocks,
                                                # In case there are only linear layers
                                                conv_kernel_sizes[0::2] if len(conv_kernel_sizes) > 0 else 1,
                                                self.params.ACTIVATION_BIT_WIDTH,
                                                self.params.PE_ROWS,
                                                self.params.SUBSECTION_SIZE if in_subsection_mode else -1
                                            )

                                            all_input_tensors = [input_tensor]

                                            # Only layers with kernel size != 1 increase the receptive field exponentially
                                            num_exponential_dilation_layers = sum([1 for i in range(0, len(conv_kernel_sizes)) if conv_kernel_sizes[i] != 1])
                                            extra_time_steps_per_sample = 1 if num_exponential_dilation_layers == 0 else 2 ** (num_exponential_dilation_layers // 2)

                                            if continuous_processing:
                                                for _ in range(num_continuous_forward_passes):
                                                    all_input_tensors.append(get_random_input_tensor_by_length(input_blocks, extra_time_steps_per_sample, self.params.ACTIVATION_BIT_WIDTH, self.params.PE_ROWS, self.params.SUBSECTION_SIZE if in_subsection_mode else -1))

                                                input_tensor = np.concatenate(all_input_tensors, axis=1)

                                            correct_out, intermediates, unscaled = tcn_network(
                                                input_tensor, quant_layers,
                                                weight_bit_width=self.params.WEIGHT_BIT_WIDTH,
                                                act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                                accum_bit_width=self.params.ACCUMULATION_BIT_WIDTH,
                                                slog2_weights=slog2_weights
                                            )

                                            correct_outs = correct_out[:, ::extra_time_steps_per_sample]
                                            unscaleds = unscaled[:, ::extra_time_steps_per_sample]

                                            all_blocks = conv_blocks + linear_blocks
                                            expected_output_channels = (-2 if send_all_argmax_chunks else -1) if classification else all_blocks[-1]*(self.params.SUBSECTION_SIZE if in_subsection_mode else self.params.PE_ROWS)

                                            for i, (input_tensor, correct_out, unscaled) in enumerate(zip(all_input_tensors, correct_outs.T, unscaleds.T)):
                                                asic_out = await self.forward(input_tensor, expected_output_channels, in_subsection_mode=in_subsection_mode, require_single_chunk=in_subsection_mode, expect_pre_post_idle_state=not continuous_processing, toggle_processing=not (continuous_processing and i != 0), activation_memory_address=activation_memory_address)

                                                if check_memory_contents:
                                                    all_kernel_sizes = conv_kernel_sizes + [1] * len(linear_blocks)

                                                    self.log_info("> Verifying memory contents")

                                                    input_memory_contents = await self.get_input_memory_contents(0, (all_kernel_sizes[0]) * input_blocks)

                                                    input_sequence = input_tensor.T.flatten().reshape(-1, self.params.PE_COLS)
                                                    expected_input_memory_contents = np.zeros((input_memory_contents.shape[0], self.params.PE_COLS), dtype=int)

                                                    # Simulate overwriting of values in the input memory
                                                    for i in range(len(input_sequence)):
                                                        expected_input_memory_contents[i % input_memory_contents.shape[0]] = input_sequence[i]

                                                    for i in range(len(input_memory_contents)):
                                                        assert np.array_equal(input_memory_contents[i], expected_input_memory_contents[i]), f"Input memory contents do not match the input tensor for input address {i}"

                                                    base_activation_address = 0

                                                    for j in range(len(intermediates)):
                                                        num_activation_rows = all_kernel_sizes[j+1] * all_blocks[j]
                                                        activation_memory_contents = await self.get_activation_memory_contents(base_activation_address, num_activation_rows)

                                                        dilation = j // 2 + 1

                                                        activation_sequence = intermediates[j]
                                                        activation_sequence = activation_sequence[::dilation]
                                                        activation_sequence = activation_sequence.T.flatten().reshape(-1, self.params.PE_COLS)

                                                        expected_activation_memory_contents = np.zeros((activation_memory_contents.shape[0], self.params.PE_COLS), dtype=int)

                                                        for i in range(len(activation_sequence)):
                                                            expected_activation_memory_contents[i % activation_memory_contents.shape[0]] = activation_sequence[i]

                                                        for i in range(len(activation_memory_contents)):
                                                            assert np.array_equal(activation_memory_contents[i], expected_activation_memory_contents[i]), f"Activation memory contents do not match the input tensor for input address {i + base_activation_address} (layer {j})"

                                                        base_activation_address += num_activation_rows

                                                assert_asic_out(asic_out, correct_out, unscaled, classification, is_new_task, send_all_argmax_chunks)

                                                # TODO: try few-shot learning with the current set of weights
                                                # TODO: try continual learning with the current set of weights

                                            if continuous_processing:
                                                await self.stop_continuous_processing()

                if in_subsection_mode:
                    await self.disable_subsection_mode()

    async def learn_with_few_shots(self,
                                    dataset,
                                    shots: int,
                                    query_shots: int,
                                    ways: int,
                                    num_batches: int,
                                    ways_for_continued_learning: int,
                                    expected_accuracies: Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
                                    quant_state_dict_file_path: str,
                                    n_last_layers_to_remove: Optional[int] = None,
                                    require_single_chunk: bool = False,
                                    seed: int = 0,
                                    l2_options: Union[str, bool] = 'both',
                                    check_memory_contents: bool = True,
                                    in_subsection_mode: bool = False,
                                    icl_shots: int = 0,
                                    are_icl_shots_labeled: bool = False,
                                    power_down_memories_while_running: tuple = (False, True),
                                    clip_inputs: bool = True,
                                    input_padding_strategy: Literal["pre", "post"] = 'pre',
                                    pre_embed_hook: Optional[Callable] = None,
                                    verify: bool = True,
                                    expected_out: Optional[List[List[int]]] = None,
                                    send_all_argmax_chunks: Optional[bool] = None):
        set_seed(seed)

        in_context_learning = icl_shots != 0
        output_blocks_for_icl = 0
        labels_for_icl = None

        if in_context_learning:
            assert ways == 1, "In-context learning only works with 1 way"
            assert ways_for_continued_learning == 0, "In-context learning only works with 0 ways for continued learning"

            output_blocks_for_icl = 1

        if send_all_argmax_chunks is None:
            send_all_argmax_chunks = ways > 2**self.params.HIGH_SPEED_OUT_PINS
        elif send_all_argmax_chunks == False and ways > 2**self.params.HIGH_SPEED_OUT_PINS:
            raise ValueError("Number of ways is too large for the given number of high speed output pins. Enable `send_all_argmax_chunks` to send all argmax chunks instead of just the first one.")

        if ways_for_continued_learning == 0:
            if in_context_learning:
                quant_in, quant_layers = get_quant_state_dict_and_layers(quant_state_dict_file_path, True,
                                                                scale_bit_width=self.params.SCALE_BIT_WIDTH,
                                                                accepted_layers=None,
                                                                n_last_layers_to_remove=n_last_layers_to_remove)
                
                icl_embedding_size = quant_layers[-1][-1][0][1].shape[0]
                icl_block_size = math.ceil(icl_embedding_size / (self.params.SUBSECTION_SIZE if in_subsection_mode else self.params.PE_ROWS))

                new_in_size = (icl_shots+1) * icl_block_size

                if are_icl_shots_labeled:
                    new_in_size += icl_shots * icl_block_size

                # TODO: move ICL network creation outside of FSL verification function
                num_icl_net_layers = 3
                quant_layers_icl = get_random_tcn(
                    new_in_size,
                    [],
                    [],
                    [2, 3, output_blocks_for_icl],
                    pe_rows=self.params.PE_ROWS,
                    weight_bit_width=self.params.WEIGHT_BIT_WIDTH,
                    act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                    bias_bit_width=self.params.BIAS_BIT_WIDTH,
                    subsection_size=self.params.SUBSECTION_SIZE if in_subsection_mode else -1,
                    slog2_weights=True
                )

                net_cfg = await self.write_network_to_asic(quant_layers + quant_layers_icl, padding_value=0, subsection_network=in_subsection_mode, icl_layers_shots=(num_icl_net_layers, icl_shots), are_icl_shots_labeled=are_icl_shots_labeled)

                if are_icl_shots_labeled:
                    from chameleon.core.net_transfer_utils import activations_to_messages

                    max_activation_value = 2**self.params.ACTIVATION_BIT_WIDTH-1
                    labels_for_icl = np.ones((icl_embedding_size*icl_shots, 1), dtype=int) * max_activation_value

                    all_input_messages = activations_to_messages(labels_for_icl,
                            activation_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                            in_subsection_mode=in_subsection_mode,
                            pe_array_size=self.params.PE_ROWS,
                            subsection_size=self.params.SUBSECTION_SIZE,
                            spi_message_bit_width=self.params.MESSAGE_BIT_WIDTH)
                    
                    label_start_address = net_cfg["blocks_per_layer_times_kernel_size_cumsum"][-num_icl_net_layers-1] - icl_block_size * icl_shots
                    message_activation_row_factor = (self.params.ACTIVATION_BIT_WIDTH * self.params.PE_ROWS) // self.params.MESSAGE_BIT_WIDTH

                    await self.write_asic_memory_over_spi("activations", all_input_messages, label_start_address*message_activation_row_factor)
            else:
                quant_in, quant_layers, net_cfg = await self.write_quant_state_dict_to_asic(
                    quant_state_dict_file_path,
                    True, None, n_last_layers_to_remove, padding_value=0,
                    subsection_network=in_subsection_mode, continued_learning=False
                )
        else:
            quant_in, quant_layers = get_quant_state_dict_and_layers(quant_state_dict_file_path, True,
                                                                    scale_bit_width=self.params.SCALE_BIT_WIDTH,
                                                                    accepted_layers=None,
                                                                    n_last_layers_to_remove=n_last_layers_to_remove)
            net_cfg = None

        embedding_size = get_output_size(quant_layers)

        # TODO: TEST LEARNING STOPPING!!!!!
        
        # Set a fixed random seed for Numpy as the FewShot class used Numpy's random number generation to create few-shot learning batches
        set_seed(seed)

        few_shot_data = FewShot(dataset, ways, shots, query_shots)
        few_shot_data_iter = iter(few_shot_data)

        first_loop = True
        called_pre_embed = False

        # Make batches deterministic across runs
        batches = []

        for _ in range(num_batches):
            batches.append(next(few_shot_data_iter))

        if l2_options == 'both':
            l2_options = (False, True)
        else:
            l2_options = (l2_options,)

        classifications = [True, False] if in_context_learning else [True]

        results = []

        # Using L2 distance or not is in the outer loop as it changes the accuracy result
        for use_l2_for_few_shot in l2_options:
            prev_accuracy = []

            # Add here all processing variations to check!
            for power_down_memories_while_running_setting in power_down_memories_while_running:
                for classification in classifications:
                    self.log_info(f"> Running with: \n - shots: {shots}\n - icl_shots: {icl_shots}\n - in_subsection_mode: {in_subsection_mode}\n - power_down_memories_while_running: {power_down_memories_while_running}\n - classification: {classification}\n - use_l2_for_few_shot: {use_l2_for_few_shot}\n - ways_for_continued_learning: {ways_for_continued_learning}\n - in_context_learning: {in_context_learning}")

                    # reset clock to normal

                    few_shot_weight_scale, few_shot_bias_scale = await self.configure_processing_setup(
                        classification=classification,
                        power_down_memories_while_running=power_down_memories_while_running_setting,
                        require_single_chunk=require_single_chunk,
                        shots=1 + icl_shots if in_context_learning else shots, use_l2_for_few_shot=use_l2_for_few_shot,
                        continued_learning=ways_for_continued_learning != 0,
                        in_subsection_mode=in_subsection_mode,
                        power_down_small_bias=in_subsection_mode, # TODO THIS IS NOT ALWAYS TRUE!
                        in_context_learning=in_context_learning,
                        send_all_argmax_chunks=send_all_argmax_chunks
                    )

                    results.append({"settings": {
                    "power_down_memories_while_running": power_down_memories_while_running_setting,
                    "classification": classification,
                    "use_l2_for_few_shot": use_l2_for_few_shot},
                    "results_per_batch": []})

                    iterable = range(num_batches)

                    if not self.verbose:
                        iterable = tqdm(iterable, desc="Processing batches", unit="batch", total=num_batches)

                    for batch_idx in iterable:
                        if self.verbose:
                            self.log_info(f"> ({batch_idx+1}/{num_batches}) Processing batch")

                        # Reset learned ways if we learn for a second time
                        if first_loop == True:
                            first_loop = False
                        else:
                            await self.reset_learned_ways()

                        ((X_support, y_support), (X_query, y_query)) = batches[batch_idx]

                        ways_embeds = []

                        iterable = range(ways)

                        if not self.verbose:
                            iterable = tqdm(iterable, desc="Training ways", unit="way", total=ways, leave=False)

                        for way in iterable:
                            if self.verbose:
                                self.log_info(f"> ({way+1}/{ways}) Training way")

                            creating_continued_output_layer = way < ways_for_continued_learning

                            embds = []

                            for i in range(shots):
                                is_new_task = i == 0

                                sample_idx = way*shots+i
                                assert y_support[sample_idx] == way, "Support set is not correctly labeled"
                                x = get_quant_input(X_support[sample_idx].numpy(), quant_layers, quant_in, clip=clip_inputs, pad=input_padding_strategy)

                                if verify and not expected_out:
                                    emb, _, _ = tcn_network(x, quant_layers,
                                                                    act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                                                    accum_bit_width=self.params.ACCUMULATION_BIT_WIDTH)
                                else:
                                    emb = None
                                
                                embds.append(emb)

                                if not creating_continued_output_layer:
                                    expect_out = i >= icl_shots and in_context_learning

                                    if expect_out:
                                        if classification:
                                            expected_output_channels = -1
                                        else:
                                            expected_output_channels = output_blocks_for_icl * self.params.PE_COLS
                                    else:
                                        expected_output_channels = 0

                                    if pre_embed_hook is not None and not called_pre_embed:
                                        await pre_embed_hook()
                                        called_pre_embed = True

                                    asic_out = await self.forward(x, require_single_chunk=require_single_chunk,
                                                    expected_output_channels=expected_output_channels, is_new_task=is_new_task,
                                                    in_subsection_mode=in_subsection_mode)
                                    
                                    if i >= icl_shots and in_context_learning:
                                        if i == icl_shots:
                                            self.log_info("Starting ICL shot testing")

                                        # Put last embedding first as the last embedding is written
                                        # inside the chip to the first location in memory for the next layer
                                        embds = embds[-1:] + embds[:-1]

                                        embds_array = np.array(embds).flatten()

                                        if labels_for_icl is not None:
                                            embds_array = np.concatenate((embds_array, labels_for_icl.flatten())).flatten()

                                        correct_out, _, unscaled = tcn_network(embds_array, quant_layers_icl,
                                        act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                        accum_bit_width=self.params.ACCUMULATION_BIT_WIDTH)

                                        assert_asic_out(asic_out, correct_out.flatten(), unscaled, classification, is_new_task, send_all_argmax_chunks)

                                        # Remove first embedding
                                        embds = embds[1:]

                            if in_context_learning:
                                break

                            if verify and not expected_out:
                                embds = np.array(embds)
                                sum_embds = np.sum(embds, axis=0)
                                shift_sum_embds = left_right_shift(sum_embds, few_shot_weight_scale)

                                assert np.max(shift_sum_embds) <= 2**(2**(self.params.WEIGHT_BIT_WIDTH-1)-1), "Embeddings are too large for the given bit width. Decrease the FEW_SHOT_SCALE value."

                                shift_sum_embds = np.where(shift_sum_embds == 0, 1, shift_sum_embds)
                                log2s = np.log2(shift_sum_embds)
                                flog2_sum_embds = np.floor(log2s)
                                correct_embeds = flog2_sum_embds.flatten().astype(int)
                            else:
                                correct_embeds = None

                            ways_embeds.append(correct_embeds)

                            if way + 1 == ways_for_continued_learning:
                                weight, bias = compute_expected_weight_and_bias(ways_embeds, self.params.WEIGHT_BIT_WIDTH, few_shot_bias_scale, use_l2_for_few_shot)

                                weight = int_to_slog2(weight, self.params.WEIGHT_BIT_WIDTH)
                                net_cfg = await self.write_network_to_asic(quant_layers + [(weight, bias, -1)], padding_value=0, subsection_network=in_subsection_mode, continued_learning=True)
                            elif not creating_continued_output_layer and check_memory_contents:
                                self.log_info("> Verifying memory contents")

                                true_rows = self.params.SUBSECTION_SIZE if in_subsection_mode else self.params.PE_ROWS

                                embedding_blocks = math.ceil(embedding_size/true_rows)
                                ways_blocks = math.ceil((way+1-ways_for_continued_learning)/true_rows)

                                weight_mem_cont = await self.get_weight_memory_contents(net_cfg["max_weight_address"], embedding_blocks*ways_blocks)

                                asic_embeds = []

                                if in_subsection_mode:
                                    on_chip_embedding_size = (embedding_size // self.params.SUBSECTION_SIZE) * self.params.PE_ROWS
                                else:
                                    on_chip_embedding_size = embedding_size
                    
                                for i in range(ways_blocks):
                                    asic_embeds.append(weight_mem_cont[i*embedding_blocks:(i+1)*embedding_blocks, :, :].reshape(on_chip_embedding_size, -1))

                                asic_embeds = np.concatenate(asic_embeds, axis=1)

                                if in_subsection_mode:
                                    asic_embeds = get_subsection_blocks_2d(asic_embeds, self.params.SUBSECTION_SIZE, self.params.PE_ROWS)

                                assert np.max(asic_embeds) < 2**(self.params.WEIGHT_BIT_WIDTH-1), "Embeddings are too large for the given bit width. Decrease the FEW_SHOT_SCALE value."

                                npt.assert_array_equal(np.array(ways_embeds).T[:, ways_for_continued_learning:], asic_embeds[:, :way+1-ways_for_continued_learning], "Embeddings are not correct")
                                assert np.all(asic_embeds[:, way+1-ways_for_continued_learning:] == 0), "Memory is not zeroed out after the last way"

                                bias_mem_cont = await self.get_bias_memory_contents(net_cfg["max_bias_address"], ways_blocks)
                                bias_mem_cont = bias_mem_cont.flatten()

                                if in_subsection_mode:
                                    bias_mem_cont = get_subsection_blocks_1d(bias_mem_cont, self.params.SUBSECTION_SIZE, self.params.PE_ROWS)

                                _, expected_bias = compute_expected_weight_and_bias(ways_embeds, self.params.WEIGHT_BIT_WIDTH, few_shot_bias_scale, use_l2_for_few_shot, self.params.FEW_SHOT_ACCUMULATION_BIT_WIDTH)

                                npt.assert_array_equal(bias_mem_cont[:way+1-ways_for_continued_learning], expected_bias[ways_for_continued_learning:], "Biases are not correct")
                                assert np.all(bias_mem_cont[way+1-ways_for_continued_learning:] == -2**(self.params.BIAS_BIT_WIDTH-1)), "All biases for unlearned ways should be -2**(self.params.BIAS_BIT_WIDTH-1)"

                        if not in_context_learning:
                            num_correct = 0

                            if verify and not expected_out:
                                ways_arr, bias = compute_expected_weight_and_bias(ways_embeds, self.params.WEIGHT_BIT_WIDTH, few_shot_bias_scale, use_l2_for_few_shot)

                            testing_results = []

                            iterable = enumerate(zip(X_query, y_query))

                            if not self.verbose:
                                iterable = tqdm(iterable, desc="Testing query samples", unit="sample", total=query_shots*ways, leave=False)

                            for i, (X_query_sample, y_query_sample) in enumerate(zip(X_query, y_query)):
                                if self.verbose:
                                    self.log_info(f"> ({i+1}/{query_shots*ways}) Testing query sample")

                                # Take scalar so that when doing += on y_query_sample doesnt change y_query
                                # in place so that the next time y_query is used, it's wrong
                                y_query_sample = y_query_sample.item()

                                x = get_quant_input(X_query_sample.numpy(), quant_layers, quant_in, clip=clip_inputs, pad=input_padding_strategy)

                                offset = math.ceil(ways_for_continued_learning / self.params.PE_COLS) * self.params.PE_COLS - ways_for_continued_learning

                                if ways_for_continued_learning > 0:
                                    y_query_sample += (y_query_sample >= ways_for_continued_learning) * offset

                                correct_out = None
                                unscaled = None

                                if verify:
                                    if expected_out is None:
                                        correct_emb_out, _, _ = tcn_network(x, quant_layers,
                                                                        act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                                                        accum_bit_width=self.params.ACCUMULATION_BIT_WIDTH)
                                        
                                        correct_out, unscaled = fc_bias_relu(correct_emb_out, ways_arr, bias, -1, -1, 2**(self.params.ACCUMULATION_BIT_WIDTH-1))
                                        correct_out = correct_out.flatten()
                                        correct_out_argmax = np.argmax(unscaled)

                                        if ways_for_continued_learning > 0:
                                            correct_out_argmax += (correct_out_argmax >= ways_for_continued_learning) * offset
                                    else:
                                        correct_out_argmax = expected_out[batch_idx][i]
                                        correct_out = np.zeros((embedding_size,), dtype=int)
                                        correct_out[correct_out_argmax] = 1
                                else:
                                    correct_out_argmax = None

                                asic_out = await self.forward(x, require_single_chunk=require_single_chunk, in_subsection_mode=in_subsection_mode, expected_output_channels=-2 if send_all_argmax_chunks else -1)

                                if verify:
                                    assert_asic_out(asic_out, correct_out, unscaled, True, False, send_all_argmax_chunks)

                                    asic_out_argmax = asic_out[0] if type(asic_out) is tuple else asic_out

                                    # Check if ASIC output results in a correct classification
                                    num_correct += asic_out_argmax == y_query_sample

                                testing_results.append((asic_out, y_query_sample))

                            accuracy = num_correct / len(y_query)

                            results[-1]["results_per_batch"].append((testing_results, accuracy))

                            if expected_accuracies[use_l2_for_few_shot] is not None:
                                assert accuracy > expected_accuracies[use_l2_for_few_shot][0], f"Accuracy is too low: {accuracy}"

                            self.log_info(f"> Accuracy: {accuracy*100}%")

                            if len(prev_accuracy) == num_batches:
                                assert accuracy == prev_accuracy[batch_idx], f"Accuracy is not consistent: {accuracy} != {prev_accuracy[batch_idx]}"
                            else:
                                prev_accuracy.append(accuracy)

                    if not in_context_learning:
                        avg_accuracy = np.mean(prev_accuracy)

                        if self.verbose:
                            self.log_info(f"> Average accuracy: {avg_accuracy*100}%")

                        if expected_accuracies[use_l2_for_few_shot] is not None:
                            assert avg_accuracy > expected_accuracies[use_l2_for_few_shot][1], f"Average accuracy is too low: {np.mean(prev_accuracy)}"

        return results

    async def classify_dataset(self,
                               dataset: Dataset,
                               net_path_or_state_dict: Union[str, Path, OrderedDict],
                               n_samples_to_test: Optional[int] = None,
                               write_state_dict_to_asic: bool = True,
                               configure_processing_setup: bool = True,
                               require_single_chunk: bool = False,
                               fill_input_memory: bool = False,
                               power_down_memories_while_running: bool = False,
                               in_subsection_mode: bool = False,
                               accepted_layers: Optional[List[str]] = None,
                               clip_inputs: bool = True,
                               input_padding_strategy: Literal["pre", "post"] = "pre",
                               padding_value: Optional[int] = None,
                               expected_min_accuracy: Optional[float] = None,
                               activation_memory_address: Optional[int] = None,
                               send_all_argmax_chunks: Optional[bool] = None,
                               pre_classify_hook: Optional[Callable] = None,
                               verify: bool = True):
        if write_state_dict_to_asic:
            quant_in, quant_layers, _ = await self.write_quant_state_dict_to_asic(
                net_path_or_state_dict,
                True, accepted_layers, padding_value=padding_value,
                subsection_network=in_subsection_mode,
                activation_memory_address=activation_memory_address
            )
        else:
            quant_in, quant_layers = get_quant_state_dict_and_layers(
                net_path_or_state_dict, True,
                scale_bit_width=self.params.SCALE_BIT_WIDTH,
                accepted_layers=accepted_layers
            )

        last_layer_output_size = get_output_size(quant_layers)

        if send_all_argmax_chunks is None:
            send_all_argmax_chunks = last_layer_output_size > 2**self.params.HIGH_SPEED_OUT_PINS
        elif send_all_argmax_chunks == False and last_layer_output_size > 2**self.params.HIGH_SPEED_OUT_PINS:
            raise ValueError("Number of ways is too large for the given number of high speed output pins. Enable `send_all_argmax_chunks` to send all argmax chunks instead of just the first one.")

        if configure_processing_setup:
            await self.configure_processing_setup(
                classification=True,
                fill_input_memory=fill_input_memory,
                power_down_memories_while_running=power_down_memories_while_running,
                require_single_chunk=require_single_chunk,
                in_subsection_mode=in_subsection_mode,
                power_down_small_bias=in_subsection_mode, # todo this should be a separate flag
                load_inputs_from_activation_memory=activation_memory_address is not None,
                send_all_argmax_chunks=send_all_argmax_chunks
            )

        called_pre_classify = False
        num_correct = 0
        results = []

        if n_samples_to_test is None:
            n_samples_to_test = len(dataset)

        iterable = range(n_samples_to_test)

        if not self.verbose:
            iterable = tqdm(iterable, desc="Testing samples", unit="sample", total=n_samples_to_test)

        for i in iterable:
            x, y = dataset[i]

            if type(x) == list:
                x = np.array(x)
            elif type(x) != np.ndarray:
                x = x.numpy()

            x = get_quant_input(x, quant_layers, quant_in, clip=clip_inputs, pad=input_padding_strategy)

            if pre_classify_hook and not called_pre_classify:
                pre_classify_hook()

                called_pre_classify = True

            asic_out = await self.forward(
                x, require_single_chunk=require_single_chunk,
                in_subsection_mode=in_subsection_mode,
                activation_memory_address=activation_memory_address,
                send_all_argmax_chunks=-2 if send_all_argmax_chunks else -1
            )

            if verify:
                scaled_out, _, unscaled_out = tcn_network(x, quant_layers,
                                                act_bit_width=self.params.ACTIVATION_BIT_WIDTH,
                                                accum_bit_width=self.params.ACCUMULATION_BIT_WIDTH,
                                                slog2_weights=True)
                
                assert_asic_out(asic_out, scaled_out, unscaled_out, True, False, send_all_argmax_chunks)

            asic_out_argmax = asic_out[0] if type(asic_out) is tuple else asic_out
            num_correct += asic_out_argmax == y

            if not self.verbose:
                iterable.set_postfix(acc=f"{(num_correct/(i+1)):.4f}")
                
            results.append((asic_out, y))

        accuracy = num_correct / n_samples_to_test

        if expected_min_accuracy is not None:
            assert accuracy > expected_min_accuracy, f"Accuracy is too low: {accuracy}"

        if self.verbose:
            self.log_info(f"Accuracy: {accuracy*100}%")

        return accuracy, results
