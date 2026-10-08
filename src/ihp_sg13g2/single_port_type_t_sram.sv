`ifndef __IHP_SG13G2_SINGLE_PORT_TYPE_T_SRAM_SV__
`define __IHP_SG13G2_SINGLE_PORT_TYPE_T_SRAM_SV__

`include "ihp_sg13g2_sram_macro.sv"

/**
 * Drop-in replacement for the behavioral single_port_type_t_sram from asic-cells
 * (same module name, parameters and ports), built from open IHP SG13G2 SRAM macros.
 *
 * The smallest macro depth that fits NUM_ROWS is used, and macros are tiled along
 * the width. Unused macro bits are never written and their outputs are dropped.
 *
 * Difference with the behavioral model: during a write, Q keeps its previous value
 * instead of loading the old contents of the written row.
 */
module single_port_type_t_sram #(
    parameter integer WIDTH = 128,
    parameter integer NUM_ROWS = 4096,
    localparam integer AddressWidth = $clog2(NUM_ROWS)
) (
    // Global inputs
    input CLK,  // Clock (synchronous read/write)

    // Control and data inputs
    input CEB,  // Chip enable, active low
    input WEB,  // Write enable: WEB is low for writing; for reading, WEB is high
    input [AddressWidth-1:0] A,  // Address bus
    input [WIDTH-1:0] D,  // Data input bus (write)
    input [WIDTH-1:0] M,  // Mask bus (overwite = 0, otherwise = 1)

    // Data output
    output [WIDTH-1:0] Q  // Data output bus (read)
);

    localparam int MacroRows = NUM_ROWS <= 256 ? 256 : NUM_ROWS <= 512 ? 512 : NUM_ROWS <= 1024 ? 1024 : 2048;
    localparam int MacroWidth = (MacroRows == 2048 || WIDTH % 64 == 0) ? 64 : 32;
    localparam int MacroAddressWidth = $clog2(MacroRows);
    localparam int NumMacros = (WIDTH + MacroWidth - 1) / MacroWidth;
    localparam int PaddedWidth = NumMacros * MacroWidth;

    if (NUM_ROWS > 2048) begin : gen_raise_error__too_many_rows_for_ihp_sg13g2_1p_sram
        $fatal(1, "ERROR: NUM_ROWS (%0d) is larger than the deepest IHP SG13G2 single-port SRAM macro used (2048)", NUM_ROWS);
    end

    wire [MacroAddressWidth-1:0] macro_address = MacroAddressWidth'(A);
    wire [PaddedWidth-1:0] padded_data_in = PaddedWidth'(D);
    wire [PaddedWidth-1:0] padded_bit_mask = PaddedWidth'(~M);
    wire [PaddedWidth-1:0] padded_data_out;

    for (genvar i = 0; i < NumMacros; i = i + 1) begin : gen_macros
        ihp_sg13g2_sram_1p_macro #(
            .ROWS (MacroRows),
            .WIDTH(MacroWidth)
        ) macro (
            .clk(CLK),
            .memory_enable(~CEB),
            .write_enable(~CEB & ~WEB),
            .read_enable(~CEB & WEB),
            .address(macro_address),
            .data_in(padded_data_in[i*MacroWidth+:MacroWidth]),
            .bit_mask(padded_bit_mask[i*MacroWidth+:MacroWidth]),
            .data_out(padded_data_out[i*MacroWidth+:MacroWidth])
        );
    end

    assign Q = padded_data_out[WIDTH-1:0];

endmodule

`endif
