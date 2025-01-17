`timescale 1ns / 1ns

module and_gate_tb;
    // Declare inputs as reg and outputs as wire
    reg a;
    reg b;
    wire y;

    // Instantiate the AND gate module
    and_gate uut (
        .a(a), 
        .b(b), 
        .y(y)
    );

    // Testbench logic
    initial begin
        // Monitor output
        $monitor("Time=%0t | a=%b b=%b | y=%b", $time, a, b, y);

        // Test cases
        a = 0; b = 0; #10;  // Wait 10 time units
        a = 0; b = 1; #10;
        a = 1; b = 0; #10;
        a = 1; b = 1; #10;

        // End simulation
        $finish;
    end
endmodule
