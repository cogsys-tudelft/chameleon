import numpy as np

import cocotb
from cocotb.triggers import Timer


NUM_TEST_CASES = 1000
RANDOM_SEED = 4


@cocotb.test()
async def test_argmax_tree(dut):
    np.random.seed(RANDOM_SEED)

    WIDTH = int(dut.WIDTH)
    N = int(dut.N)

    for i in range(NUM_TEST_CASES+1):
        if i == 0:
            # Test that the argmax module always returns the first index when multiple values are the same
            data = np.zeros(N, dtype=int)
        else:
            # Generate a random sequence of values of length N and width WIDTH
            data = np.random.randint(-2**(WIDTH-1), 2**(WIDTH-1), N, dtype=int)

        dut.data.value = data.tolist()
        
        await Timer(1, units='ns')

        expected_argmax = np.argmax(data)
        actual_argmax = int(dut.argmax.value)

        # Check the result
        assert actual_argmax == expected_argmax, f"Expected index: {expected_argmax}, Got: {actual_argmax}"
