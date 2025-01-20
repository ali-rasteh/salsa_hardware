/*
Data is always a stream of values occassionally reserved in reuse buffers.
Differen scenarios for the movement buffer
- Could contain values of one set of weights to be applied on the input, in this case
the uppest cell receives the whole set of weights and each cell pops its weight out of it.
- Could contain the values of calculated outputs from all above cell to be spit out at the most bottom cell.
*/



`include "mult_IEEE754.v"
`include "add_IEEE754.v"




module sa_cell #(parameter DATA_WIDTH = 32, MODE_WIDTH=1, CTRL_WIDTH = 3, MOVE_BUFF_DEPTH=16) (
    input clk,
    input rst,
    input [MODE_WIDTH-1:0] mode,
    input [CTRL_WIDTH-1:0] ctrl,
    input [DATA_WIDTH-1:0] left,
    input [DATA_WIDTH-1:0] up,
    input [DATA_WIDTH-1:0] move_buff_in,
    input move_buff_in_valid,

    output reg [DATA_WIDTH-1:0] right,
    output reg [DATA_WIDTH-1:0] down,
    output reg [DATA_WIDTH-1:0] move_buff_out,
    output reg move_buff_out_valid
);

    parameter REUSE_BUFF_DEPTH = 2;
    parameter RESULT_BUFF_DEPTH = 2;
    parameter REAL_OFFSET = 0;
    parameter IMAG_OFFSET = 1;
    parameter STATE_DEPTH = 4;




    reg [DATA_WIDTH-1:0] reuse_buff [REUSE_BUFF_DEPTH-1:0];
    reg [DATA_WIDTH-1:0] move_buff [MOVE_BUFF_DEPTH-1:0];
    reg [DATA_WIDTH-1:0] result_buff [RESULT_BUFF_DEPTH-1:0];

    reg [$clog2(STATE_DEPTH)-1:0] state;
    reg [$clog2(MOVE_BUFF_DEPTH)-1:0] move_buff_push_index;
    reg [$clog2(MOVE_BUFF_DEPTH)-1:0] move_buff_pop_index;


    reg [DATA_WIDTH-1:0] mult_in_1;
    reg [DATA_WIDTH-1:0] mult_in_2;
    wire [DATA_WIDTH-1:0] mult_out;
    reg [DATA_WIDTH-1:0] add_in_1;
    reg [DATA_WIDTH-1:0] add_in_2;
    wire [DATA_WIDTH-1:0] add_out;

    integer i;




    mult_IEEE754_32bit #(.WIDTH(DATA_WIDTH)) mult_inst (
        .a(mult_in_1),
        .b(mult_in_2),
        .product(mult_out)
    );

    adder_IEEE754_32bit #(.WIDTH(DATA_WIDTH)) add_inst (
        .a(add_in_1),
        .b(add_in_2),
        .sum(add_out)
    );





    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < REUSE_BUFF_DEPTH; i = i + 1) begin
                reuse_buff[i] <= 0;
            end
            for (i = 0; i < MOVE_BUFF_DEPTH; i = i + 1) begin
                move_buff[i] <= 0;
            end
            for (i = 0; i < RESULT_BUFF_DEPTH; i = i + 1) begin
                result_buff[i] <= 0;
            end

            state <= 0;
            move_buff_push_index <= 0;
            move_buff_pop_index <= 0;
            mult_in_1 <= 0;
            mult_in_2 <= 0;
            add_in_1 <= 0;
            add_in_2 <= 0;

            right <= 0;
            down <= 0;
            move_buff_out <= 0;
            move_buff_out_valid <= 0;



        end else begin
            
            right <= mult_in_1;
            down <= add_out;

            if (state == 0) begin
                reuse_buff[0] <= left;
                mult_in_1 <= left;
                mult_in_2 <= move_buff[move_buff_pop_index];
                state <= 1;
            end else if (state == 1) begin
                reuse_buff[1] <= left;
                mult_in_1 <= left;
                mult_in_2 <= move_buff[move_buff_pop_index];
                move_buff_pop_index <= move_buff_pop_index + 1;
                state <= 2;
            end else if (state == 2) begin
                mult_in_1 <= reuse_buff[0];
                mult_in_2 <= move_buff[move_buff_pop_index];
                state <= 3;
            end else if (state == 3) begin
                mult_in_1 <= reuse_buff[1];
                mult_in_2 <= move_buff[move_buff_pop_index];
                move_buff_pop_index <= move_buff_pop_index - 1;
                state <= 0;
            end



            if (mode == 0) begin
                // Code for the accumulation mode of the spatial array
                add_in_1 <= mult_out;
                add_in_2 <= up;



            end else if (mode == 1) begin
                // Code for the element-wise mode of the spatial array

                if (state == 0) begin
                    result_buff[0] <= mult_out;
                end else if (state == 1) begin
                    result_buff[1] <= mult_out;
                end else if (state == 2) begin
                    add_in_1 <= mult_out;
                    add_in_2 <= result_buff[1];
                end else if (state == 3) begin
                    add_in_1 <= {~mult_out[31], mult_out[30:0]};
                    add_in_2 <= result_buff[0];
                end

                if (move_buff_in_valid == 1) begin
                    move_buff[move_buff_push_index] <= move_buff_in;
                    move_buff_push_index <= move_buff_push_index + 1;
                    if (move_buff_push_index == MOVE_BUFF_DEPTH) begin
                        move_buff_push_index <= 0;
                    end
                end

            end



        end

    end

    
endmodule
