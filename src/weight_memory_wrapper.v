module weight_memory_wrapper #(
    parameter WIDTH = 1024,
    parameter NUM_ROWS = 128,
    parameter SRAM1_WIDTH = 64,
    parameter SRAM2_WIDTH = 128,
    parameter SRAM1_OFFSET = 256,
    localparam ADDRESS_WIDTH = $clog2(NUM_ROWS)
) (
    input clk,
    input rst,

    input [ADDRESS_WIDTH-1:0] address,
    input [WIDTH-1:0] data_in,
    input [WIDTH-1:0] mask,
    input chip_select,
    input write_enable,
    input power_down,

    input in_4x4_mode,

    output [WIDTH-1:0] data_out
);
    wire chip_enable;
    wire invert_write_enable;
    wire [WIDTH-1:0] bit_write_enable;

    // "Chip enable, active low for SRAM operation; active high for fuse data setting"
    assign chip_enable = ~chip_select;

    // For writing, write is low; for reading, write is high
    assign invert_write_enable = ~write_enable;

    // For writing, M is 0; for reading, M can be anything ("active low" for writing)
    assign bit_write_enable = ~mask;

    wire [SRAM1_OFFSET-SRAM2_WIDTH-1:0] data_out_4x4;

    reg address_msb_delayed;

    always @(posedge clk) begin
        if (rst) begin
            address_msb_delayed <= 1'b0;
        end else if (in_4x4_mode & chip_select) begin
            address_msb_delayed <= address[ADDRESS_WIDTH-1];
        end
    end

    assign data_out[SRAM1_OFFSET-SRAM2_WIDTH-1:0] = in_4x4_mode ? {{SRAM1_WIDTH{1'b0}}, (address_msb_delayed == 1'b0 ? data_out_4x4[SRAM1_WIDTH-1:0] : data_out_4x4[2*SRAM1_WIDTH-1:SRAM1_WIDTH])} : data_out_4x4;

    genvar i;
    generate
        for (i = 0; i < 2; i = i + 1) begin
            wire select_this_memory = address[ADDRESS_WIDTH-1] == i;
            wire use_single_port = in_4x4_mode & select_this_memory;

`ifndef SYNTHESIZE_FOR_SILICON
            single_port_type_t_sram weight_memory_64_inst (
`else
            real_weight_memory_64 weight_memory_64_inst (
`endif
                .CLK(clk),
                .CEB(in_4x4_mode ? ~(select_this_memory & chip_select) : chip_enable),
                .WEB(in_4x4_mode ? ~(select_this_memory & write_enable) : invert_write_enable),
                .A(address[ADDRESS_WIDTH-2:0]),
                .D(use_single_port ? data_in[SRAM1_WIDTH-1:0] : data_in[SRAM1_WIDTH+(i*SRAM1_WIDTH)-1-:SRAM1_WIDTH]),
                .M(use_single_port ? bit_write_enable[SRAM1_WIDTH-1:0] : bit_write_enable[SRAM1_WIDTH+(i*SRAM1_WIDTH)-1-:SRAM1_WIDTH]),
                .Q(data_out_4x4[SRAM1_WIDTH+(i*SRAM1_WIDTH)-1-:SRAM1_WIDTH])
`ifdef SYNTHESIZE_FOR_SILICON
                        .PD(power_down),
`endif
            );
        end
    endgenerate

    genvar j;
    generate
        for (j = 0; j < 7; j = j + 1) begin
`ifndef SYNTHESIZE_FOR_SILICON
            single_port_type_t_sram weight_memory_128_inst (
`else
            real_weight_memory_128 weight_memory_128_inst (
`endif
                .CLK(clk),
                .CEB(chip_enable),
                .WEB(invert_write_enable),
                .A(address[ADDRESS_WIDTH-2:0]),
                .D(data_in[SRAM1_OFFSET+(j*SRAM2_WIDTH)-1-:SRAM2_WIDTH]),
                .M(bit_write_enable[SRAM1_OFFSET+(j*SRAM2_WIDTH)-1-:SRAM2_WIDTH]),
                .Q(data_out[SRAM1_OFFSET+(j*SRAM2_WIDTH)-1-:SRAM2_WIDTH])
`ifdef SYNTHESIZE_FOR_SILICON
                        .PD(power_down),
`endif
            );
        end
    endgenerate

endmodule
