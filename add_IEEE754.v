module fp_adder_32bit #(parameter WIDTH = 32) (
    input  [WIDTH-1:0] a,    // WIDTH-bit floating-point input a
    input  [WIDTH-1:0] b,    // WIDTH-bit floating-point input b
    output [WIDTH-1:0] sum   // WIDTH-bit floating-point sum
);
    // Extract sign, exponent, and mantissa
    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];
    wire [7:0] exp_b = b[30:23];
    wire [23:0] mant_a = {1'b1, a[22:0]};
    wire [23:0] mant_b = {1'b1, b[22:0]};

    // Align the exponents
    wire [7:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [23:0] mant_a_shifted = (exp_a > exp_b) ? mant_a : (mant_a >> exp_diff);
    wire [23:0] mant_b_shifted = (exp_b > exp_a) ? mant_b : (mant_b >> exp_diff);
    wire [7:0] exp_sum = (exp_a > exp_b) ? exp_a : exp_b;

    // Add or subtract the mantissas based on the signs
    wire [24:0] mant_sum;
    if (sign_a == sign_b) begin
        mant_sum = mant_a_shifted + mant_b_shifted;
    end else begin
        mant_sum = (mant_a_shifted > mant_b_shifted) ?
                   (mant_a_shifted - mant_b_shifted) :
                   (mant_b_shifted - mant_a_shifted);
    end

    // Normalize the result
    wire [22:0] mant_normalized;
    wire [7:0] exp_normalized;
    if (mant_sum[24]) begin
        mant_normalized = mant_sum[23:1];
        exp_normalized = exp_sum + 1;
    end else begin
        mant_normalized = mant_sum[22:0];
        exp_normalized = exp_sum;
    end

    // Determine the sign of the result
    wire sign_sum = (mant_a_shifted > mant_b_shifted) ? sign_a : sign_b;

    // Combine the normalized values into the final sum
    assign sum = {sign_sum, exp_normalized, mant_normalized};

endmodule
