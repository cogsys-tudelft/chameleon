from pathlib import Path
from typing import Union, Optional

from pynq import Overlay

from pymeasure.instruments.keysight import KeysightE36312A

from basic_asic_fpga_bridge import AsicFpgaBridge

from chameleon.core.shared_utils import clog2
from chameleon.core.chameleon_interface import ChameleonInterface, ChameleonParams


class ChameleonFpgaBridge(AsicFpgaBridge, ChameleonInterface):
    def __init__(self, overlay: Overlay,
                 config_memory_file: Union[str, Path],
                 pointer_file: Union[str, Path],
                 launch_current_measurement_unit: bool = True):
        """Initialize the Chameleon Bridge.

        Args:
            overlay (Overlay): The Overlay object initialized from the bitstream file that was built on top of the basic-asic-fpga-bridge project.
            config_memory_file (Union[str, Path]): Path to Chameleon configuration memory config file
            pointer_file (Union[str, Path]): Path to Chameleon pointer config file
        """

        params = ChameleonParams(
            MESSAGE_BIT_WIDTH=32,
            CODE_BIT_WIDTH=4,
            START_ADDRESS_BIT_WIDTH=16,
            ACTIVATION_BIT_WIDTH=4,
            SCALE_BIT_WIDTH=4,
            BIAS_BIT_WIDTH=14,
            PE_COLS=16,
            PE_ROWS=16,
            SUBSECTION_SIZE=4,
            WEIGHT_BIT_WIDTH=4,
            HIGH_SPEED_IN_PINS=16,
            HIGH_SPEED_OUT_PINS=8,
            ACTIVATION_ROWS=256,
            WEIGHT_ROWS=1024,
            BIAS_ROWS=128,
            INPUT_ROWS=32,
            ACCUMULATION_BIT_WIDTH=18,
            MAX_KERNEL_SIZE=15,
            MAX_NUM_CHANNELS=1024,
            FEW_SHOT_ACCUMULATION_BIT_WIDTH=18,
            MAX_SHOTS=127,
            MAX_NUM_LOGITS=1024,
            WAIT_CYCLES_WIDTH=4
        )

        memory_map = {
            "weights": {
                "num_rows": 512,
                "bit_width": 1024
            },
            "biases": {
                "num_rows": 128,
                "bit_width": 256
            },
            "activations": {
                "num_rows": 256,
                "bit_width": 64
            },
            "inputs": {
                "num_rows": 32,
                "bit_width": 64
            }
        }

        bram_bit_width = 32
        bridge_input_bram_size = 2048
        bridge_output_bram_size = 8192

        connect_string_E363312A = "TCPIP::192.168.3.3::inst0::INSTR"

        num_generated_clocks = 5

        max_bridge_input_message_bit_width = clog2(bridge_input_bram_size*bram_bit_width//params.HIGH_SPEED_OUT_PINS)
        max_bridge_output_message_bit_width = clog2(bridge_output_bram_size*bram_bit_width//params.HIGH_SPEED_IN_PINS)

        fpga_config_layout = [
            ("fpga_rst", 1),
            ("ext_clk_selection", clog2(num_generated_clocks + 1)),
            ("custom_clock_generation_cycles", 19),
            ("expected_bridge_input_data_transfers", max_bridge_input_message_bit_width),
            ("required_bridge_output_data_transfers", max_bridge_output_message_bit_width),
        ]

        fpga_state_layout = [
            ("done_receving", 1),
            ("completed_bridge_input_data_transfers", max_bridge_input_message_bit_width),
            ("completed_bridge_output_data_transfers", max_bridge_output_message_bit_width)
        ]
    
        asic_inputs_layout = [
            ("rst_async", 1),
            ("enable_clk_int", 1),
            ("toggle_processing", 1),
            ("is_new_task", 1),
        ]

        asic_outputs_layout = [
            ("in_idle", 1),
        ]

        asic_inputs_config = {
            "num_rows": bridge_output_bram_size,
            "row_width": bram_bit_width,
            "entry_width": params.HIGH_SPEED_IN_PINS
        
        }

        asic_outputs_config = {
            "num_rows": bridge_input_bram_size,
            "row_width": bram_bit_width,
            "entry_width": params.HIGH_SPEED_OUT_PINS
        }

        self.supply = KeysightE36312A(connect_string_E363312A)

        AsicFpgaBridge.__init__(
            self,
            overlay=overlay,
            config_layout=fpga_config_layout,
            state_layout=fpga_state_layout,
            inputs_layout=asic_inputs_layout,
            outputs_layout=asic_outputs_layout,
            num_generated_clocks=num_generated_clocks,
            asic_inputs_config=asic_inputs_config,
            asic_outputs_config=asic_outputs_config)

        ChameleonInterface.__init__(
            self,
            params=params,
            memory_map=memory_map,
            config_memory_file=config_memory_file,
            pointer_file=pointer_file,
            verbose=True)
            

    def enable_supply(self, core: Optional[bool] = None, macro: Optional[bool] = None, io: Optional[bool] = None):
        if core is not None:
            self.supply.ch_1.output_enabled = core
        
        if macro is not None:
            self.supply.ch_2.output_enabled = macro

        if io is not None:
            self.supply.ch_3.output_enabled = io

    def configure_voltage_setpoints(self, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3):
        if core_voltage > 1.1:
            raise ValueError("Core voltage can never be higher than 1.1V!")
        
        if macro_voltage > 1.1:
            raise ValueError("Macro voltage can never be higher than 1.1V!")

        if io_voltage > 3.3:
            raise ValueError("IO voltage can never be higher than 3.3V due to ASIC limiations!")
        elif io_voltage < 1.1:
            raise ValueError("IO voltage can never be lower than 1.1V due to level-shifter limitations!")

        self.supply.ch_1.voltage_setpoint = core_voltage
        self.supply.ch_2.voltage_setpoint = macro_voltage
        self.supply.ch_3.voltage_setpoint = io_voltage

    async def switch_to_internal_asic_clock(self, ring_oscillator_stage_selection: int,
                                      enable_clock_divider: bool = False,
                                      check_back: bool = True,
                                      timeout: float = 0.1):
        """Switch the clock that the ASIC runs on from the clock coming from the host to the internal ASIC clock.

        Args:
            ring_oscillator_stage_selection (int): The stage of the ring oscillator to be used for the internal ASIC clock.
            enable_clock_divider (bool, optional): Whether to enable the clock divider to present the divided clock on the `clk_int_div` pin. Defaults to False.
            check_back (bool, optional): Whether to check if the chip has returned to the idle state after the clock switch. Defaults to True.
            timeout (float, optional): Waiting time between the various switching steps. Defaults to 0.1.
        """
        
        # Configure internal ASIC clock via SPI
        await self.send_asic_config_over_spi({"ring_oscillator_stage_selection": ring_oscillator_stage_selection})
        await self.sleep(timeout)

        # Disable the clock going from host to ASIC
        self.set_fpga_config(ext_clk_selection=0)
        await self.sleep(timeout)

        # Enable internal ASIC clock
        self.set_asic_inputs(enable_clk_int=1)
        await self.sleep(timeout)

        # Reset ASIC
        await self.reset_asic(timeout)

        if check_back:
            assert self.get_asic_outputs("in_idle"), "ASIC is not in idle state after switching; clock switching failed"

        await self.send_asic_config_over_spi({"enable_clock_divider": enable_clock_divider})
