// 16-bit IEEE 754 (binary16) multiplier
// - Assumes inputs are finite & normalized (no NaN/Inf/subnormals)
// - Rounding: round-to-nearest, ties-to-even
// - Overflow -> max finite; Underflow -> 0

`timescale 1ns / 1ns

module mult_IEEE754_16bit #(parameter WIDTH = 16) (
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output [WIDTH-1:0] product
);
    // Field extraction
    wire        sa = a[15];
    wire  [4:0] ea = a[14:10];
    wire  [9:0] fa = a[9:0];

    wire        sb = b[15];
    wire  [4:0] eb = b[14:10];
    wire  [9:0] fb = b[9:0];

    // Sign of result
    wire s = sa ^ sb;

    // Add hidden 1 (normalized numbers only)
    wire [10:0] ma = {1'b1, fa}; // 1.xxx -> 11 bits
    wire [10:0] mb = {1'b1, fb};

    // 11x11 -> 22-bit product
    wire [21:0] prod = ma * mb;

    // Normalization: product is in [1.0, 4.0)
    wire        prod_msb = prod[21];   // 1 if >= 2.0
    // Exponent math with bias (bias = 15)
    // Add 1 to exponent if we normalize by shifting right when prod_msb==1
    wire [6:0] exp_unrounded = ea + eb - 7'd15 + {6'b0, prod_msb};

    // Take the top bits for fraction before rounding:
    // Keep 10 fraction bits; gather guard and sticky for RNE
    wire [9:0] frac_pre  = prod_msb ? prod[20:11] : prod[19:10]; // 10 bits to round
    wire       guard_bit = prod_msb ? prod[10]    : prod[9];
    wire       sticky    = prod_msb ? (|prod[9:0]) : (|prod[8:0]);

    // Round-to-nearest, ties-to-even
    wire       lsb       = frac_pre[0];
    wire       round_up  = guard_bit & (sticky | lsb);
    wire [10:0] frac_rounded_wcarry = {1'b0, frac_pre} + {10'b0, round_up};

    // If rounding overflows the 10-bit fraction, renormalize (shift right 1) and bump exponent
    wire       frac_carry = frac_rounded_wcarry[10];
    wire [9:0] frac_final = frac_carry ? frac_rounded_wcarry[10:1] : frac_rounded_wcarry[9:0];

    // Final exponent after rounding carry
    wire [7:0] exp_final_wide = exp_unrounded + {7'b0, frac_carry};

    // Handle overflow/underflow (simple clamp)
    // Overflow: exponent >= 31 -> set to max finite (exp=30, frac=all 1s)
    // Underflow: exponent <= 0  -> return zero
    wire overflow  = (exp_final_wide >= 8'd31);
    wire underflow = (exp_final_wide[7] || (exp_final_wide == 8'd0)); // negative or zero

    wire [4:0] exp_final = overflow  ? 5'd30 :
                           underflow ? 5'd0  :
                                      exp_final_wide[4:0];

    wire [9:0] frac_out  = overflow  ? 10'h3FF :
                           underflow ? 10'h000 :
                                      frac_final;

    assign product = {s, exp_final, frac_out};
endmodule
