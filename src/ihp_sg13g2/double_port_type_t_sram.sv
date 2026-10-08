`ifndef __IHP_SG13G2_DOUBLE_PORT_TYPE_T_SRAM_SV__
`define __IHP_SG13G2_DOUBLE_PORT_TYPE_T_SRAM_SV__

`include "ihp_sg13g2_sram_macro.sv"

/**
 * Drop-in replacement for the behavioral double_port_type_t_sram from asic-cells
 * (same module name, parameters and ports), built from open IHP SG13G2 dual-port
 * SRAM macros. Port A of each macro is used for writing and port B for reading.
 *
 * The smallest macro depth that fits NUM_ROWS is used, and macros are tiled along
 * the width. Unused macro bits are never written and their outputs are dropped.
 */
module double_port_type_t_sram #(
    parameter integer WIDTH = 128,
    parameter integer NUM_ROWS = 4096,
    localparam integer AddressWidth = $clog2(NUM_ROWS)
) (
    // Global inputs
    input CLK,  // Clock (synchronous read/write)

    // Control and data inputs
    input REB,  // Read enable, active low
    input WEB,  // Write enable: WEB is low for writing; for reading, WEB is high
    input [AddressWidth-1:0] AA,  // Address bus (write)
    input [AddressWidth-1:0] AB,  // Address bus (read)
    input [WIDTH-1:0] D,  // Data input bus (write)
    input [WIDTH-1:0] M,  // Mask bus (overwrite = 0, otherwise = 1)

    // Data output
    output [WIDTH-1:0] Q  // Data output bus (read)
);

    localparam int MacroRows = NUM_ROWS <= 256 ? 256 : NUM_ROWS <= 512 ? 512 : 1024;
    localparam int MacroWidth = 32;
    localparam int MacroAddressWidth = $clog2(MacroRows);
    localparam int NumMacros = (WIDTH + MacroWidth - 1) / MacroWidth;
    localparam int PaddedWidth = NumMacros * MacroWidth;

    if (NUM_ROWS > 1024) begin : gen_raise_error__too_many_rows_for_ihp_sg13g2_2p_sram
        $fatal(1, "ERROR: NUM_ROWS (%0d) is larger than the deepest IHP SG13G2 dual-port SRAM macro (1024)", NUM_ROWS);
    end

    wire [MacroAddressWidth-1:0] macro_write_address = MacroAddressWidth'(AA);
    wire [MacroAddressWidth-1:0] macro_read_address = MacroAddressWidth'(AB);
    wire [PaddedWidth-1:0] padded_data_in = PaddedWidth'(D);
    wire [PaddedWidth-1:0] padded_bit_mask = PaddedWidth'(~M);
    wire [PaddedWidth-1:0] padded_data_out;

    for (genvar i = 0; i < NumMacros; i = i + 1) begin : gen_macros
        ihp_sg13g2_sram_2p_macro #(
            .ROWS (MacroRows),
            .WIDTH(MacroWidth)
        ) macro (
            .clk(CLK),

            .write_enable(~WEB),
            .write_address(macro_write_address),
            .data_in(padded_data_in[i*MacroWidth+:MacroWidth]),
            .bit_mask(padded_bit_mask[i*MacroWidth+:MacroWidth]),

            .read_enable(~REB),
            .read_address(macro_read_address),
            .data_out(padded_data_out[i*MacroWidth+:MacroWidth])
        );
    end

    assign Q = padded_data_out[WIDTH-1:0];

endmodule

`endif
