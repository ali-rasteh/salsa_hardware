/*
Data is always a stream of values occassionally reserved in reuse buffers.
Differen scenarios for the movement buffer
- Could contain values of one set of weights to be applied on the input, in this case
the uppest cell receives the whole set of weights and each cell pops its weight out of it.
- Could contain the values of calculated outputs from all above cell to be spit out at the most bottom cell.
*/

`include "mult_IEEE754_16.v"
`include "add_IEEE754_16.v"
`include "mux.v"

`timescale 1ns / 1ns


module sa_cell #(
    parameter DATA_WIDTH = 16,
    parameter CTRL_WIDTH = 9,
    parameter WD_BUFFER_DEPTH = 16,
    parameter INPUT_BUFFER_DEPTH = 4,
    parameter PARTIALS_BUFFER_DEPTH = 2)
(
    input wire clk,
    input wire rst,

    // input [MODE_WIDTH-1:0] mode,        // 0 for accumulation mode, 1 for element-wise mode
    // input complex_mode,                 // 0 for real, 1 for complex
    input wire [CTRL_WIDTH-1:0] ctrl,
    input wire [DATA_WIDTH-1:0] left_in,
    input wire [DATA_WIDTH-1:0] result_in,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire data_in_valid,

    input wire [$clog2(WD_BUFFER_DEPTH)-1:0] wd_buffer_pop_index,
    input wire [$clog2(INPUT_BUFFER_DEPTH)-1:0] input_buffer_pop_index,
    input wire [$clog2(PARTIALS_BUFFER_DEPTH)-1:0] partials_buffer_pop_index,
    input wire add_sub,               // 0 for addition, 1 for subtraction

    output reg [DATA_WIDTH-1:0] right_out,
    output reg [DATA_WIDTH-1:0] result_out,
    output reg [DATA_WIDTH-1:0] data_out,
    output reg data_out_valid
);

    reg [DATA_WIDTH-1:0] input_buffer       [INPUT_BUFFER_DEPTH-1:0];
    reg [DATA_WIDTH-1:0] wd_buffer          [WD_BUFFER_DEPTH-1:0];
    reg [DATA_WIDTH-1:0] partials_buffer    [PARTIALS_BUFFER_DEPTH-1:0];

    reg [$clog2(WD_BUFFER_DEPTH)-1:0]       wd_buffer_push_index;
    reg [$clog2(INPUT_BUFFER_DEPTH)-1:0]    input_buffer_push_index;
    reg [$clog2(PARTIALS_BUFFER_DEPTH)-1:0] partials_buffer_push_index;

    wire wd_buffer_in_valid;
    wire input_buffer_in_valid;
    wire partials_buffer_in_valid;

    wire [DATA_WIDTH-1:0] mult_in_1;
    wire [DATA_WIDTH-1:0] mult_in_2;
    wire [DATA_WIDTH-1:0] mult_out;
    wire [DATA_WIDTH-1:0] add_in_1;
    wire [DATA_WIDTH-1:0] add_in_2;
    wire [DATA_WIDTH-1:0] add_out;
    wire [DATA_WIDTH-1:0] data_input_out;
    wire [DATA_WIDTH-1:0] mux_partials_in_wire;
    wire [DATA_WIDTH-1:0] mux_input_buffer_wire;

    wire mux_1_sel;
    wire mux_2_sel;
    wire mux_3_sel;
    wire mux_4_sel;
    wire [1:0] mux_5_sel;

    integer i;


    assign mux_1_sel                = ctrl[0];
    assign mux_2_sel                = ctrl[1];
    assign mux_3_sel                = ctrl[2];
    assign mux_4_sel                = ctrl[3];
    assign mux_5_sel                = ctrl[5:4];
    assign wd_buffer_in_valid       = ctrl[6];
    assign input_buffer_in_valid    = ctrl[7];
    assign partials_buffer_in_valid = ctrl[8];

    assign add_in_1 = mult_out;
    assign mult_in_1 = input_buffer[input_buffer_pop_index];


    mult_IEEE754_16bit #(.WIDTH(DATA_WIDTH)) mult_inst (
        .a(mult_in_1),
        .b(mult_in_2),
        .product(mult_out)
    );
    adder_IEEE754_16bit #(.WIDTH(DATA_WIDTH)) add_inst (
        .a(add_in_1),
        .b(add_in_2),
        .sub(add_sub),
        .sum(add_out)
    );

    mux #(.DATA_WIDTH(DATA_WIDTH), .N(2)) mux_1 (
        .data_in({left_in, data_in}),
        .sel(mux_1_sel),
        .data_out(mux_input_buffer_wire)
    );
    mux #(.DATA_WIDTH(DATA_WIDTH), .N(2)) mux_2 (
        .data_in({wd_buffer[wd_buffer_pop_index], data_in}),
        .sel(mux_2_sel),
        .data_out(mult_in_2)
    );
    mux #(.DATA_WIDTH(DATA_WIDTH), .N(2)) mux_3 (
        .data_in({mult_out, result_in}),
        .sel(mux_3_sel),
        .data_out(mux_partials_in_wire)
    );
    mux #(.DATA_WIDTH(DATA_WIDTH), .N(2)) mux_4 (
        .data_in({partials_buffer[partials_buffer_pop_index], result_in}),
        .sel(mux_4_sel),
        .data_out(add_in_2)
    );
    mux #(.DATA_WIDTH(DATA_WIDTH), .N(4)) mux_5 (
        .data_in({mult_in_1, mult_out, wd_buffer[wd_buffer_pop_index], add_out}),
        .sel(mux_5_sel),
        .data_out(data_input_out)
    );

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < INPUT_BUFFER_DEPTH; i = i + 1) begin
                input_buffer[i] <= {DATA_WIDTH{1'b0}};
            end
            for (i = 0; i < WD_BUFFER_DEPTH; i = i + 1) begin
                wd_buffer[i] <= {DATA_WIDTH{1'b0}};
            end
            for (i = 0; i < PARTIALS_BUFFER_DEPTH; i = i + 1) begin
                partials_buffer[i] <= {DATA_WIDTH{1'b0}};
            end

            wd_buffer_push_index <= '0;
            partials_buffer_push_index <= '0;
            input_buffer_push_index <= '0;

            right_out <= '0;
            result_out <= '0;
            data_out <= '0;
            data_out_valid <= 1'b0;

        end else begin

            right_out <= data_input_out;
            result_out <= add_out;
            data_out <= data_input_out;
            data_out_valid <= data_in_valid;

            if (data_in_valid == 1 && wd_buffer_in_valid == 1) begin
                wd_buffer[wd_buffer_push_index] <= data_in;
                wd_buffer_push_index <= (wd_buffer_push_index == WD_BUFFER_DEPTH-1)
                                        ? '0
                                        : wd_buffer_push_index + 1'b1;
            end
            
            if (input_buffer_in_valid == 1) begin
                input_buffer[input_buffer_push_index] <= mux_input_buffer_wire;
                input_buffer_push_index <= (input_buffer_push_index == INPUT_BUFFER_DEPTH-1)
                                           ? '0
                                           : input_buffer_push_index + 1'b1;
            end

            if (partials_buffer_in_valid == 1) begin
                partials_buffer[partials_buffer_push_index] <= mux_partials_in_wire;
                partials_buffer_push_index <= (partials_buffer_push_index == PARTIALS_BUFFER_DEPTH-1)
                                              ? '0
                                              : partials_buffer_push_index + 1'b1;
            end

        end
    end


endmodule
