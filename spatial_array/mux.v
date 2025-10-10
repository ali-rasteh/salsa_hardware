`timescale 1ns / 1ns

module mux #(
    parameter DATA_WIDTH = 32,
    parameter N = 2
)(
    input  wire [DATA_WIDTH*N-1:0] data_in,   // All inputs concatenated
    input  wire [$clog2(N)-1:0]    sel,       // Select line
    output wire [DATA_WIDTH-1:0]   data_out   // Selected output
);
    assign data_out = data_in[sel*DATA_WIDTH +: DATA_WIDTH];
endmodule
