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
    ACCUMULATION_BIT_WIDTH = int(dut.ACCUMULATION_BIT_WIDTH)
    COLS = int(dut.COLS)
    ROWS = int(dut.ROWS)
    CLAMP = int(dut.CLAMP)
    SCALE_BIT_WIDTH = int(dut.SCALE_BIT_WIDTH)

    max_steps = 64

    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await FallingEdge(dut.clk)

    for apply_identity in (False, True):
        for use_subsection in (False, True):
            # Test against different input block sizes (where one block is ROWS number of channel)
            for current_steps in range(3, max_steps + 1):
                inputs = np.random.randint(0, 2**ACTIVATION_BIT_WIDTH, (current_steps, COLS))
                scale_input = np.random.randint(0, 2, (current_steps,)) # Not always apply an input scale factor
                # Use min() to avoid having too large scales that result in too large outputs
                input_scales = scale_input * np.random.randint(0, min(2**SCALE_BIT_WIDTH, 4), current_steps)

                if apply_identity:
                    # Replace every weight matrix with an identity matrix
                    weights = np.stack([np.eye(COLS, ROWS, dtype=int) for _ in range(current_steps)])
                else:
                    weights = np.random.randint(0, 2**WEIGHT_BIT_WIDTH, (current_steps, ROWS, COLS))

                slog2_weights = slog2_to_int(weights, WEIGHT_BIT_WIDTH)

                # Set all inputs and weights that are not in the subsection to zero
                if use_subsection:
                    inputs[:, SUBSECTION_SIZE:] = 0

                    slog2_weights[:, SUBSECTION_SIZE:, SUBSECTION_SIZE:] = 0
                    slog2_weights[:, SUBSECTION_SIZE:, :] = 0
                    slog2_weights[:, :, SUBSECTION_SIZE:] = 0

                # Stack the weights along the first dimension to prepare for fc
                if apply_identity:
                    # Dont use slog2 weights as they transform the zero weights into non-zero weights
                    weights_format_for_fc = np.vstack(weights).T
                else: 
                    weights_format_for_fc = np.vstack(slog2_weights).T

                fc_out = fc((inputs.reshape(-1, 1).T*2**(np.repeat(input_scales, ROWS))).T, weights_format_for_fc).flatten()

                # Compute a custom lower bound on the biases to make sure that they do not dominate the result
                bias_bound = min(np.abs(fc_out).max() // 2, 2**(BIAS_BIT_WIDTH - 1)-1)
                biases = np.random.randint(-bias_bound, bias_bound, COLS)

                # Set all biases outside the subsection to zero
                if use_subsection:
                    biases[SUBSECTION_SIZE:] = 0

                fc_out = fc_out + biases

                assert fc_out.max() < 2**(ACCUMULATION_BIT_WIDTH - 1), "Accumulation bit width is too small to hold the result of the FC operation"
                assert fc_out.min() >= -2**(ACCUMULATION_BIT_WIDTH - 1), "Accumulation bit width is too small to hold the result of the FC operation"

                print(f"FC out: {fc_out}")
                print(f"Biases: {biases}")

                relu = np.maximum(fc_out, 0)
                max_val = relu.max()

                if max_val == 0:
                    out_scale = 0
                else:
                    # Compute out_scale based on maximum positive output value to avoid overflow
                    out_scale = math.ceil(math.log2(max_val)) - ACTIVATION_BIT_WIDTH

                    # When clamp is enabled, we randomly increase the out_scale to make sure more overflows happen that should be clamped
                    if CLAMP:
                        out_scale -= np.random.randint(0, 3)

                    if out_scale < 0:
                        out_scale = 0

                    assert out_scale <= (2**SCALE_BIT_WIDTH - 1), f"Out scale {out_scale} is too large to fit in {SCALE_BIT_WIDTH} bits"

                correct_out = np.right_shift(relu, out_scale)

                if not CLAMP:
                    correct_out = (correct_out & (2**ACTIVATION_BIT_WIDTH - 1))
                else:
                    # Perform overflow correction
                    correct_out = np.clip(
                        correct_out,
                        0,
                        2**ACTIVATION_BIT_WIDTH - 1
                    )
                
                correct_out = correct_out.tolist()

                # Configure PE array to be ready
                dut.use_subsection.value = use_subsection
                dut.enable.value = True
                dut.apply_identity.value = apply_identity

                # This code starts to run after the first falling edge, before the second rising edge
                for step in range(current_steps):
                    getattr(dut, 'in').value = inputs[step].tolist()
                    dut.weights.value = weights[step].flatten().tolist()

                    if scale_input[step]:
                        dut.in_scale.value = int(input_scales[step])
                        dut.apply_in_scale.value = True
                    else:
                        dut.apply_in_scale.value = False

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
