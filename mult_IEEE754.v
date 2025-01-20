module mult_IEEE754_32bit #(parameter WIDTH = 32) (
    input  [WIDTH-1:0] a,    // WIDTH-bit floating-point input a
    input  [WIDTH-1:0] b,    // WIDTH-bit floating-point input b
    output [WIDTH-1:0] product // WIDTH-bit floating-point product
);
    // Extract sign, exponent, and mantissa
    wire sign_a = a[31];
    wire sign_b = b[31];
    wire [7:0] exp_a = a[30:23];
    wire [7:0] exp_b = b[30:23];
    wire [23:0] mant_a = {1'b1, a[22:0]};
    wire [23:0] mant_b = {1'b1, b[22:0]};

    // Compute sign, exponent, and mantissa for the product
    wire sign_product = sign_a ^ sign_b;
    wire [47:0] mant_product = mant_a * mant_b;
    wire [7:0] exp_product = exp_a + exp_b - 127; // Bias adjustment

    // Normalize the product
    reg [22:0] mant_normalized;
    reg [7:0] exp_normalized;

    always @(*) begin
        if (mant_product[47]) begin
            mant_normalized = mant_product[46:24];
            exp_normalized = exp_product + 1;
        end else begin
            mant_normalized = mant_product[45:23];
            exp_normalized = exp_product;
        end
    end

    // Combine the normalized values into the final product
    assign product = {sign_product, exp_normalized, mant_normalized};

endmodule
