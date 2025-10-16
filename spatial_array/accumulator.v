`timescale 1ns / 1ns

module accumulator_IEEE754_16bit #(
    parameter WIDTH = 16
)(
    input  wire                  clk,
    input  wire                  reset,   // Active-high reset
    input  wire                  enable,  // Enable accumulation
    input  wire [WIDTH-1:0]      in,       // Input value
    output reg  [WIDTH-1:0]      sum_out  // Output accumulated sum
);

    // Internal signals
    reg  [WIDTH-1:0] sum_reg;            // Registered running sum
    wire [WIDTH-1:0] sum_next;           // Next sum from adder

    // Instantiate IEEE754 adder
    adder_IEEE754_16bit #(.WIDTH(WIDTH)) adder_inst (
        .a(sum_reg),
        .b(in),
        .sub(1'b0),          // Always add (not subtract)
        .sum(sum_next)
    );

    // Sequential logic for accumulation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            sum_reg <= {WIDTH{1'b0}};    // Reset sum to 0
        end else if (enable) begin
            sum_reg <= sum_next;         // Accumulate when enabled
        end else begin
            sum_reg <= sum_reg;          // Hold current sum when not enabled
        end
    end

    // Drive output
    always @(*) begin
        if (reset) begin
            sum_out = {WIDTH{1'b0}};
        end else if (enable) begin
            sum_out = sum_reg;
        end else begin
            sum_out = in;
        end
    end

endmodule
