`ifndef __IHP_SG13G2_SRAM_MACRO_SV__
`define __IHP_SG13G2_SRAM_MACRO_SV__

/**
 * Uniform wrappers around the open IHP SG13G2 SRAM macros (RM_IHPSG13_*), so that
 * the macro can be selected with parameters instead of by module name.
 *
 * All macros have active-high enables, a registered output that holds its value
 * when not reading, and a bit mask where 1 means "write this bit". The BIST ports
 * are tied off and A_DLY is tied high (default read timing).
 */

module ihp_sg13g2_sram_1p_macro #(
    parameter int ROWS  = 512,
    parameter int WIDTH = 64,
    localparam int AddressWidth = $clog2(ROWS)
) (
    input                     clk,
    input                     memory_enable,
    input                     write_enable,
    input                     read_enable,
    input  [AddressWidth-1:0] address,
    input  [       WIDTH-1:0] data_in,
    input  [       WIDTH-1:0] bit_mask,
    output [       WIDTH-1:0] data_out
);

`define IHP_SG13G2_SRAM_1P_PORTS \
        .A_CLK(clk), \
        .A_MEN(memory_enable), \
        .A_WEN(write_enable), \
        .A_REN(read_enable), \
        .A_ADDR(address), \
        .A_DIN(data_in), \
        .A_DLY(1'b1), \
        .A_DOUT(data_out), \
        .A_BM(bit_mask), \
        .A_BIST_CLK(1'b0), \
        .A_BIST_EN(1'b0), \
        .A_BIST_MEN(1'b0), \
        .A_BIST_WEN(1'b0), \
        .A_BIST_REN(1'b0), \
        .A_BIST_ADDR('0), \
        .A_BIST_DIN('0), \
        .A_BIST_BM('0)

    if (ROWS == 256 && WIDTH == 32) begin : gen_256x32
        RM_IHPSG13_1P_256x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 256 && WIDTH == 64) begin : gen_256x64
        RM_IHPSG13_1P_256x64_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 512 && WIDTH == 32) begin : gen_512x32
        RM_IHPSG13_1P_512x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 512 && WIDTH == 64) begin : gen_512x64
        RM_IHPSG13_1P_512x64_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 1024 && WIDTH == 32) begin : gen_1024x32
        RM_IHPSG13_1P_1024x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 1024 && WIDTH == 64) begin : gen_1024x64
        RM_IHPSG13_1P_1024x64_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else if (ROWS == 2048 && WIDTH == 64) begin : gen_2048x64
        RM_IHPSG13_1P_2048x64_c2_bm_bist sram (`IHP_SG13G2_SRAM_1P_PORTS);
    end else begin : gen_raise_error__unsupported_ihp_sg13g2_1p_sram_size
        $fatal(1, "ERROR: No IHP SG13G2 single-port SRAM macro of %0dx%0d", ROWS, WIDTH);
    end

`undef IHP_SG13G2_SRAM_1P_PORTS

endmodule

/**
 * Dual-port macro used as one write port (A) and one read port (B).
 */
module ihp_sg13g2_sram_2p_macro #(
    parameter int ROWS  = 256,
    parameter int WIDTH = 32,
    localparam int AddressWidth = $clog2(ROWS)
) (
    input clk,

    input                     write_enable,
    input  [AddressWidth-1:0] write_address,
    input  [       WIDTH-1:0] data_in,
    input  [       WIDTH-1:0] bit_mask,

    input                     read_enable,
    input  [AddressWidth-1:0] read_address,
    output [       WIDTH-1:0] data_out
);

`define IHP_SG13G2_SRAM_2P_PORTS \
        .A_CLK(clk), \
        .A_MEN(write_enable), \
        .A_WEN(write_enable), \
        .A_REN(1'b0), \
        .A_ADDR(write_address), \
        .A_DIN(data_in), \
        .A_DLY(1'b1), \
        .A_DOUT(), \
        .A_BM(bit_mask), \
        .A_BIST_CLK(1'b0), \
        .A_BIST_EN(1'b0), \
        .A_BIST_MEN(1'b0), \
        .A_BIST_WEN(1'b0), \
        .A_BIST_REN(1'b0), \
        .A_BIST_ADDR('0), \
        .A_BIST_DIN('0), \
        .A_BIST_BM('0), \
        .B_CLK(clk), \
        .B_MEN(read_enable), \
        .B_WEN(1'b0), \
        .B_REN(read_enable), \
        .B_ADDR(read_address), \
        .B_DIN('0), \
        .B_DLY(1'b1), \
        .B_DOUT(data_out), \
        .B_BM('0), \
        .B_BIST_CLK(1'b0), \
        .B_BIST_EN(1'b0), \
        .B_BIST_MEN(1'b0), \
        .B_BIST_WEN(1'b0), \
        .B_BIST_REN(1'b0), \
        .B_BIST_ADDR('0), \
        .B_BIST_DIN('0), \
        .B_BIST_BM('0)

    if (ROWS == 256 && WIDTH == 32) begin : gen_256x32
        RM_IHPSG13_2P_256x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_2P_PORTS);
    end else if (ROWS == 512 && WIDTH == 32) begin : gen_512x32
        RM_IHPSG13_2P_512x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_2P_PORTS);
    end else if (ROWS == 1024 && WIDTH == 32) begin : gen_1024x32
        RM_IHPSG13_2P_1024x32_c2_bm_bist sram (`IHP_SG13G2_SRAM_2P_PORTS);
    end else begin : gen_raise_error__unsupported_ihp_sg13g2_2p_sram_size
        $fatal(1, "ERROR: No IHP SG13G2 dual-port SRAM macro of %0dx%0d", ROWS, WIDTH);
    end

`undef IHP_SG13G2_SRAM_2P_PORTS

endmodule

`endif
