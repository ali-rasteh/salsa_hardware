`include "sa_cell.sv"

`timescale 1ns / 1ns


module sa_cell_tb;
    // Parameters
    parameter DATA_WIDTH = 16;
    parameter CTRL_WIDTH = 9;
    parameter WD_BUFFER_DEPTH = 16;
    parameter INPUT_BUFFER_DEPTH = 4;
    parameter PARTIALS_BUFFER_DEPTH = 2;

    // Inputs
    reg clk;
    reg rst;
    reg [CTRL_WIDTH-1:0] ctrl;
    reg [DATA_WIDTH-1:0] left_in;
    reg [DATA_WIDTH-1:0] result_in;
    reg [DATA_WIDTH-1:0] data_in;
    reg data_in_vali d;

    reg [$clog2(WD_BUFFER_DEPTH)-1:0] wd_buffer_pop_index;
    reg [$clog2(INPUT_BUFFER_DEPTH)-1:0] input_buffer_pop_index;
    reg [$clog2(PARTIALS_BUFFER_DEPTH)-1:0] partials_buffer_pop_index;
    reg add_sub;               // 0 for addition, 1 for subtraction

    // Outputs
    wire [DATA_WIDTH-1:0] right_out;
    wire [DATA_WIDTH-1:0] result_out;
    wire [DATA_WIDTH-1:0] data_out;
    wire data_out_valid;

    // Instantiate the sa_cell module
    sa_cell #(
        .DATA_WIDTH(DATA_WIDTH),
        .CTRL_WIDTH(CTRL_WIDTH),
        .WD_BUFFER_DEPTH(WD_BUFFER_DEPTH),
        .INPUT_BUFFER_DEPTH(INPUT_BUFFER_DEPTH),
        .PARTIALS_BUFFER_DEPTH(PARTIALS_BUFFER_DEPTH)
    ) uut (
        .clk(clk),
        .rst(rst),

        .ctrl(ctrl),
        .left_in(left_in),
        .result_in(result_in),
        .data_in(data_in),
        .data_in_valid(data_in_valid),

        .wd_buffer_pop_index(wd_buffer_pop_index),
        .input_buffer_pop_index(input_buffer_pop_index),
        .partials_buffer_pop_index(partials_buffer_pop_index),
        .add_sub(add_sub),

        .right_out(right_out),
        .result_out(result_out),
        .data_out(data_out),
        .data_out_valid(data_out_valid)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Testbench logic
    initial begin
        $monitor("clk=%0d rst=%0d ctrl=%0d left_in=%0h result_in=%0h data_in=%0h data_in_valid=%0d right_out=%0h result_out=%0h data_out=%0h data_out_valid=%0d", clk, rst, ctrl, left_in, result_in, data_in, data_in_valid, right_out, result_out, data_out, data_out_valid);
        
        // Initialize inputs
        clk = 0;
        rst = 1;

        ctrl = 0;
        left_in = 0;
        result_in = 0;
        data_in = 0;
        data_in_valid = 0;

        wd_buffer_pop_index = 0;
        input_buffer_pop_index = 0;
        partials_buffer_pop_index = 0;
        add_sub = 0;

        // Reset the design
        #10;
        rst = 0;

        // Test case 1: Accumulation mode
        left_in = 16'h3C00; // 1.0 in IEEE 754 (16-bit)
        result_in = 16'h4000; // 2.0 in IEEE 754 (16-bit)
        data_in = 16'h4200; // 3.0 in IEEE 754 (16-bit)
        data_in_valid = 1;
        #10;
        data_in_valid = 0;
        #50;

        // Test case 2: Element-wise mode
        left_in = 16'h3C00; // 1.0 in IEEE 754 (16-bit)
        result_in = 16'h4000; // 2.0 in IEEE 754 (16-bit)
        data_in = 16'h4200; // 3.0 in IEEE 754 (16-bit)
        data_in_valid = 1;
        #10;
        data_in_valid = 0;
        #50;

        // End simulation
        $finish;
    end

endmodule

