import math

import cocotb

from cocotb.triggers import FallingEdge

from cocotb.clock import Clock
import numpy as np

from chameleon.core.quant_conversions import slog2_to_int
from chameleon.core.numpy.tcn import fc


@cocotb.test()
async def test_all_ops_for_multiple_steps(dut):
    np.random.seed(2)

    WEIGHT_BIT_WIDTH = int(dut.WEIGHT_BIT_WIDTH)
    ACTIVATION_BIT_WIDTH = int(dut.ACTIVATION_BIT_WIDTH)
    BIAS_BIT_WIDTH = int(dut.BIAS_BIT_WIDTH)
    SUBSECTION_SIZE = int(dut.SUBSECTION_SIZE)
    COLS = int(dut.COLS)
    ROWS = int(dut.ROWS)

    max_steps = 64

    for apply_identity in (False,):
        for use_subsection in (False, True):
            # Test against different input block sizes (where one block is ROWS number of channel)
            for current_steps in range(1, max_steps + 1):
                inputs = np.random.randint(0, 2**ACTIVATION_BIT_WIDTH, (current_steps, COLS))
                weights = np.random.randint(0, 2**WEIGHT_BIT_WIDTH, (current_steps, ROWS, COLS))

                slog2_weights = slog2_to_int(weights, WEIGHT_BIT_WIDTH)

                # Set all inputs and weights that are not in the subsection to zero
                if use_subsection:
                    inputs[:, SUBSECTION_SIZE:] = 0
        
                    slog2_weights[:, SUBSECTION_SIZE:, SUBSECTION_SIZE:] = 0
                    slog2_weights[:, SUBSECTION_SIZE:, :] = 0
                    slog2_weights[:, :, SUBSECTION_SIZE:] = 0

                if apply_identity:
                    slog2_weights *= np.eye(ROWS, dtype=int)

                fc_out = fc(inputs.reshape(-1, 1), slog2_weights.reshape(-1, COLS)).flatten()

                # Compute a custom lower bound on the biases to make sure that they do not dominate the result
                bias_bound = min(np.abs(fc_out).max() // 2, 2**(BIAS_BIT_WIDTH-1))
                biases = np.random.randint(-bias_bound, bias_bound, COLS)

                # Set all biases outside the subsection to zero
                if use_subsection:
                    biases[SUBSECTION_SIZE:] = 0

                fc_out = fc_out + biases

                relu = np.maximum(fc_out, 0)
                max_val = relu.max()

                if max_val == 0:
                    out_scale = 0
                else:
                    # Compute out_scale based on maximum positive output value to avoid overflow
                    out_scale = math.ceil(math.log2(max_val)) - ACTIVATION_BIT_WIDTH

                    if out_scale < 0:
                        out_scale = 0

                correct_out = np.right_shift(relu, out_scale).tolist()

                cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

                await FallingEdge(dut.clk)

                # Configure PE array to be ready
                dut.use_subsection.value = use_subsection
                dut.enable.value = True
                dut.apply_identity.value = apply_identity

                # This code starts to run after the first falling edge, before the second rising edge
                for step in range(current_steps):
                    getattr(dut, 'in').value = inputs[step].tolist()
                    dut.weights.value = weights[step].flatten().tolist()

                    if step == 0:
                        dut.out_scale.value = out_scale
                        dut.biases.value = biases.tolist()

                        dut.apply_bias.value = True
                    else:
                        dut.apply_bias.value = False
                        # At this point, we dont care what the scale and biases are 
                        # anymore as they are already loaded into the PE array registers

                    await FallingEdge(dut.clk)

                assert dut.out.value == correct_out, f"Result is wrong for use_subsection={use_subsection}, apply_identity={apply_identity}, steps={current_steps}"
