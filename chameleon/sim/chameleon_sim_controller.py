from typing import Optional, List, Union
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge

from tqdm import tqdm

from chameleon.core.chameleon_interface import ChameleonInterface, ChameleonParams


class ChameleonSimController(ChameleonInterface):
    def __init__(self, dut, params: ChameleonParams, config_memory_file: Union[str, Path], pointer_file: Union[str, Path], is_padded: bool = False):
        self.dut = dut

        self.spi_half_clock = 80

        memory_sizes_and_names = {
            "weights": {
                "num_rows": params.WEIGHT_ROWS,
                "bit_width": params.WEIGHT_BIT_WIDTH * params.PE_COLS * params.PE_ROWS
            },
            "biases": {
                "num_rows": params.BIAS_ROWS,
                "bit_width": params.BIAS_BIT_WIDTH * params.PE_COLS,
            },
            "activations": {
                "num_rows": params.ACTIVATION_ROWS,
                "bit_width": params.ACTIVATION_BIT_WIDTH * params.PE_COLS
            },
            "inputs": {
                "num_rows": params.INPUT_ROWS,
                "bit_width": params.ACTIVATION_BIT_WIDTH * params.PE_COLS
            }
        }

        self.is_padded = is_padded

        super().__init__(params, memory_sizes_and_names, config_memory_file, pointer_file)

    def get_pin(self, pin_name: str):
        if self.is_padded:
            pin_name = pin_name.upper() + "_PAD"

        return getattr(self.dut, pin_name)

    def log_info(self, msg: object):
        cocotb.log.info(msg)

    async def sleep(self, duration: float):
        await Timer(duration, units='ms')

    def set_asic_inputs(self, **kwargs):
        for key, value in kwargs.items():
            self.get_pin(key).value = value

    def get_asic_outputs(self, *args):
        outputs = [self.get_pin(arg).value.integer for arg in args]

        if len(outputs) == 1:
            return outputs[0]
        
        return outputs
    
    def enable_supply(self, core: Optional[bool] = None, macro: Optional[bool] = None, io: Optional[bool] = None):
        self.log_info(f"> Skipping supply toggling for all supplies")
    
    async def transfer_spi_data(self, messages: List[str]) -> List[int]:
        responses = []

        SCK = self.get_pin("SCK")
        MOSI = self.get_pin("MOSI")
        MISO = self.get_pin("MISO")

        for message in messages:
            assert len(message) == self.params.MESSAGE_BIT_WIDTH, "Message length does not match the expected message bit width."

            response = 0

            for bit in message:
                await Timer(self.spi_half_clock, units='ns')
                SCK.value = 0
                MOSI.value = int(bit)

                await Timer(self.spi_half_clock, units='ns')
                SCK.value = 1
                response = (response << 1) | int(MISO.value)
            
            responses.append(response)

        # Set clock back to zero and leave it there
        await Timer(self.spi_half_clock, units='ns')

        # Set SPI wires to zero
        MOSI.value = 0
        SCK.value = 0

        return responses[1:]

    async def select_external_clock(self, clock: int, cycles: Optional[int] = None, timeout: float = 0.1):
        assert clock in (0, 1), "Only no clock (0) or the simulated clock (1) can be selected."

        clk_ext = self.get_pin("clk_ext")

        if clock == 1:
            assert cycles is not None, "The number of cycles must be specified when selecting the simulated clock."

            self.clk_period = 1 / cycles * 1e9

            cocotb.start_soon(Clock(clk_ext, self.clk_period, units="ns").start())
        else:
            clk_ext.value = 0

    def zero_inputs(self):
        self.set_asic_inputs(SCK=0, in_acknowledge=0)

    async def reset_host(self, timeout: float = 0.1):
        self.log_info("> Skipping host reset in simulation")
    
    async def send_high_speed_inputs(self, input_messages: List[int]):
        in_request = self.get_pin("in_request")
        out_acknowledge = self.get_pin("out_acknowledge")
        data_in = self.get_pin("data_in")

        for i, input_message in tqdm(enumerate(input_messages), desc="Sending inputs", unit="input chunk", total=len(input_messages)):
            if i == 0:
                cocotb.log.info("> Sending inputs...")

            in_request.value = 1
            data_in.value = input_message

            await RisingEdge(out_acknowledge)
            in_request.value = 0
            await FallingEdge(out_acknowledge)

    async def receive_high_speed_outputs(self, expected_num_outputs: int) -> List[int]:
        outputs = []

        out_request = self.get_pin("out_request")
        in_acknowledge = self.get_pin("in_acknowledge")
        data_out = self.get_pin("data_out")

        for _ in range(expected_num_outputs):
            await RisingEdge(out_request)

            if self.is_padded:
                # Very, very important fix that took me three days to find.
                # Sometimes, out_request goes high, but the data on the output bus needs
                # a few hundred picoseconds (~200) to be driven to the correct value
                await self.sleep(round(1e3 * (self.clk_period * 1e-9 / 25), 10))

            outputs.append(int(data_out.value))
            in_acknowledge.value = 1

            await FallingEdge(out_request)
            in_acknowledge.value = 0

        return outputs

    def configure_voltage_setpoints(self, core_voltage: float = 1.1, macro_voltage: float = 1.1, io_voltage: float = 3.3):
        self.log_info("> Skipping voltage setpoint setting in simulation")
