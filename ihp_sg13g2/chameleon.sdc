##########################################################################
###
### SDC constraints file of Chameleon on IHP SG13G2 (LibreLane/OpenSTA)
###
### Structure based on:
###     TU Delft EE4615 lecture on the automated digital IC design flow
###     March 2022, C. Frenkel
###
### Constrains the chameleon core at a 10 MHz clock:
###   - There is no pad ring, so the constraints are on the core ports.
###   - USE_RING_OSCILLATOR is undefined, so clk_int is tied to 0 in
###     src/chameleon.sv and Yosys removes the clock OR gate and the clock
###     divider. The main clock is therefore defined on the clk_ext port,
###     there are no divided clocks and clk_int_div is a constant.
###
##########################################################################

#####################################
#                                   #
#      TIMING IN ACTIVE MODE        #
#                                   #
#####################################

# Must match CLOCK_PERIOD in config.yaml, which sets the delay target of ABC during synthesis
set CLK_PERIOD         100
set MIN_SCK_PERIOD     800
set SCK_PERIOD         [expr max($CLK_PERIOD*8, $MIN_SCK_PERIOD)]

set IO_DLY             5.0

# Clock uncertainty of the IHP SG13G2 PDK (CLOCK_UNCERTAINTY_CONSTRAINT in LibreLane)
set CLK_UNCERTAINTY    $::env(CLOCK_UNCERTAINTY_CONSTRAINT)

#####################################
#                                   #
#           MAIN CLOCKS             #
#                                   #
#####################################

# Main on-chip clock (clk = clk_ext | clk_int, with clk_int tied low)
create_clock -name "clk" -period $CLK_PERIOD -waveform [list 0 [expr $CLK_PERIOD/2.0]] [get_ports clk_ext]

# SPI slave clock
create_clock -name "SCK" -period $SCK_PERIOD -waveform [list 0 [expr $SCK_PERIOD/2.0]] [get_ports SCK]

# The main clock and SCK are asynchronous: the SPI clock domain crossings are
# secured with synchronizers, so the tool does not check paths between them
set_clock_groups -asynchronous -group "clk" -group "SCK"

# Clock distribution latency and uncertainty
set_clock_uncertainty $CLK_UNCERTAINTY [all_clocks]

# LibreLane exposes OPENLANE_SDC_IDEAL_CLOCKS=1 before clock tree synthesis
if { [info exists ::env(OPENLANE_SDC_IDEAL_CLOCKS)] && $::env(OPENLANE_SDC_IDEAL_CLOCKS) } {
    unset_propagated_clock [all_clocks]
} else {
    set_propagated_clock [all_clocks]
}

#####################################
#                                   #
#         BOUNDARY CONDITIONS       #
#                                   #
#####################################

# Without pads, the outputs drive the inputs of the pad cells, so the output
# load of the PDK is used (in fF in LibreLane)
set_load [expr $::env(OUTPUT_CAP_LOAD) / 1000.0] [all_outputs]
set_input_transition 0.5 [all_inputs]

#####################################
#                                   #
#         INPUT/OUPUT DELAYS        #
#                                   #
#####################################

# Only the SPI data ports are timed. All other inputs and outputs are left
# unconstrained (no input/output delay), so they are not timed:
#   - rst_async, enable_clk_int, toggle_processing and is_new_task are
#     asynchronous inputs that are synchronized on chip, and in_idle is a
#     status output
#   - data_in, in_request, out_acknowledge, data_out, out_request and
#     in_acknowledge form the handshaked high-speed buses
#   - clk_int_div is constant, as clk_int is tied low

# Input - MOSI
set_input_delay [expr $IO_DLY]  -network_latency_included -max -clock "SCK" -clock_fall [get_ports MOSI]
set_input_delay [expr -$IO_DLY] -network_latency_included -min -clock "SCK" -clock_fall [get_ports MOSI]

# Output - MISO
set_output_delay [expr $IO_DLY]  -network_latency_included -max -clock "SCK" [get_ports MISO]
set_output_delay [expr -$IO_DLY] -network_latency_included -min -clock "SCK" [get_ports MISO]
