`include "sa_cell.v"

`timescale 1ns / 1ns


module sa_cell_tb;
    // Parameters
    parameter DATA_WIDTH = 32;
    parameter MODE_WIDTH = 1;
    parameter CTRL_WIDTH = 3;
    parameter MOVE_BUFF_DEPTH = 16;

    // Inputs
    reg clk;
    reg rst;
    reg [MODE_WIDTH-1:0] mode;
    reg [CTRL_WIDTH-1:0] ctrl;
    reg [DATA_WIDTH-1:0] left;
    reg [DATA_WIDTH-1:0] up;
    reg [DATA_WIDTH-1:0] move_buff_in;
    reg move_buff_in_valid;

    // Outputs
    wire [DATA_WIDTH-1:0] right;
    wire [DATA_WIDTH-1:0] down;
    wire [DATA_WIDTH-1:0] move_buff_out;
    wire move_buff_out_valid;

    // Instantiate the sa_cell module
    sa_cell #(
        .DATA_WIDTH(DATA_WIDTH),
        .MODE_WIDTH(MODE_WIDTH),
        .CTRL_WIDTH(CTRL_WIDTH),
        .MOVE_BUFF_DEPTH(MOVE_BUFF_DEPTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .mode(mode),
        .ctrl(ctrl),
        .left(left),
        .up(up),
        .move_buff_in(move_buff_in),
        .move_buff_in_valid(move_buff_in_valid),

        .right(right),
        .down(down),
        .move_buff_out(move_buff_out),
        .move_buff_out_valid(move_buff_out_valid)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Testbench logic
    initial begin
        $monitor("clk=%0d rst=%0d mode=%0d ctrl=%0d left=%0h up=%0h move_buff_in=%0h move_buff_in_valid=%0d right=%0h down=%0h move_buff_out=%0h move_buff_out_valid=%0d", clk, rst, mode, ctrl, left, up, move_buff_in, move_buff_in_valid, right, down, move_buff_out, move_buff_out_valid);
        
        // Initialize inputs
        clk = 0;
        rst = 1;
        mode = 0;
        ctrl = 0;
        left = 0;
        up = 0;
        move_buff_in = 0;
        move_buff_in_valid = 0;

        // Reset the design
        #10;
        rst = 0;

        // Test case 1: Accumulation mode
        mode = 0;
        left = 32'h3f800000; // 1.0 in IEEE 754
        up = 32'h40000000; // 2.0 in IEEE 754
        move_buff_in = 32'h40400000; // 3.0 in IEEE 754
        move_buff_in_valid = 1;
        #10;
        move_buff_in_valid = 0;
        #50;

        // Test case 2: Element-wise mode
        mode = 1;
        left = 32'h3f800000; // 1.0 in IEEE 754
        up = 32'h40000000; // 2.0 in IEEE 754
        move_buff_in = 32'h40400000; // 3.0 in IEEE 754
        move_buff_in_valid = 1;
        #10;
        move_buff_in_valid = 0;
        #50;

        // End simulation
        $finish;
    end

endmodule