import warnings

from pymeasure.instruments.keithley import KeithleyDMM6500


def get_mm():
    # DMM6500 does not work well over ethernet, hence we use USB
    # can find the below connect string in the sys comms > USB panel of the device (it is reported in hex there)
    connect_string_DMM6500 = "USB0::1510::25856::04537337::0::INSTR"
    connect_string_DMM6500 = "TCPIP::192.168.3.2::inst0::INSTR"


    return KeithleyDMM6500(connect_string_DMM6500)



def start_current_measurement(mm, sample_rate: float, total_seconds: float, range: float = 1e-7, first_time = False):
    sample_count = int(sample_rate * total_seconds)

    # Using f-string
    formatted = f"{range:.10f}".rstrip('0').rstrip('.')
    
    print('pre pre')
    mm.write(":DIG:FUNC 'CURR'")  
    mm.write(":TRAC:CLE 'defbuffer1'")  
    mm.write(f":TRAC:POIN {sample_count}, 'defbuffer1'")  
    mm.write(f":DIG:COUNT {sample_count}")  
    mm.write(f":DIG:CURR:RANG {formatted}")  
    mm.write(f":DIG:CURR:SRATE {sample_rate}")  
    mm.write(":TRACe:FILL:MODE ONCE, 'defbuffer1'")

    if not first_time:
        print('pre init')
        # !!!! was workign first !!!!! mm.write(":INIT")  # Ensures measurement starts immediately
        mm.write(":INIT")

        print('pre opc')
        
        # Takes a long time!
        mm.ask("*OPC?")  # Wait for operation to complete

        print('opc dnoe')
        
        # then reading all values again, also takes quite some time
        # ! was working first ! return mm.values(f":TRAC:DATA? {1}, {sample_count}")
        ignore = mm.values(f":TRAC:DATA? {1}, {sample_count}")

    print('pre meas dig')

    mm.write("MEAS:DIG?")

    # print('meas dig')

    # import time

    # time.sleep(total_seconds)

    # print('post sleep')


def get_mm_data(mm, expected_sample_count: int, max_retries: int = 10):
    for _ in range(max_retries):
        data = mm.values(f":TRAC:DATA? {1}, {expected_sample_count}")
        
        if len(data) == expected_sample_count:
            # If we were to try to get the data again from the multimeter at
            # this point, from experience is it then again the correct data

            if max(data) == 9.9e37:
                warnings.warn("Overflow happened during measurement")

            return data
        
    print(f"No valid data was returned within {max_retries} retries!")
    
    return None


def close_mm(mm, max_retries: int = 50):
    for _ in range(max_retries):
        mm.close()
