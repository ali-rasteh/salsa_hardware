module sa_cell #(parameter DATA_WIDTH = 32, CTRL_WIDTH = 3, MOVE_BUFF_DEPTH=8) (
    input clk,
    input rst,
    input [CTRL_WIDTH-1:0] ctrl,
    input [DATA_WIDTH-1:0] left,
    input [DATA_WIDTH-1:0] up,
    input [DATA_WIDTH-1:0] move_buffer_in,
    output reg [DATA_WIDTH-1:0] right,
    output reg [DATA_WIDTH-1:0] down,
    output reg [DATA_WIDTH-1:0] move_buffer_out
);
    reg [DATA_WIDTH-1:0] reuse_buffer_real;
    reg [DATA_WIDTH-1:0] reuse_buffer_imag;
    reg [DATA_WIDTH-1:0] move_buff [MOVE_BUFF_DEPTH-1:0];
    reg [DATA_WIDTH-1:0] result_buff_real;
    reg [DATA_WIDTH-1:0] result_buff_imag;


    always @(posedge clk) begin
        if (rst) begin
            right <= 0;
            down <= 0;
            move_buffer_out <= 0;
        end else begin
            right <= left;
            down <= up;
            move_buffer_out <= move_buffer_in;
        end
    end

    
endmodule
