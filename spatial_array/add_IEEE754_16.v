// Simple IEEE-754 binary16 (half-precision) adder/subtractor
// Format: [15] sign | [14:10] exponent (bias=15) | [9:0] fraction
// Assumes inputs are finite (no NaN/Inf). Rounds by truncation.

`timescale 1ns / 1ns

module adder_IEEE754_16bit #(parameter WIDTH = 16) (
    input  wire [WIDTH-1:0] a,
    input  wire [WIDTH-1:0] b,
    input  wire             sub,   // 0: a + b  |  1: a - b  (flip b's sign)
    output wire [WIDTH-1:0] sum
);
    // ---- Extract fields ----
    wire        sa = a[15];
    wire [4:0]  ea = a[14:10];
    wire [9:0]  fa = a[9:0];

    wire        sb = b[15];
    wire [4:0]  eb = b[14:10];
    wire [9:0]  fb = b[9:0];

    // Effective sign of b after add/sub select
    wire        sb_eff = sb ^ sub;

    // ---- Build significands with hidden bit (11 bits total for normal) ----
    // For subnormals (exp==0), hidden bit is 0; for normals, hidden bit is 1.
    wire [10:0] ma = (ea == 0) ? {1'b0, fa} : {1'b1, fa};
    wire [10:0] mb = (eb == 0) ? {1'b0, fb} : {1'b1, fb};

    // ---- Choose the operand with the larger "magnitude" exponent, then mantissa ----
    wire a_bigger_mag = (ea > eb) || ((ea == eb) && (ma >= mb));

    wire        sx = a_bigger_mag ? sa     : sb_eff;
    wire [4:0]  ex = a_bigger_mag ? ea     : eb;
    wire [10:0] mx = a_bigger_mag ? ma     : mb;

    wire        sy = a_bigger_mag ? sb_eff : sa;
    wire [4:0]  ey = a_bigger_mag ? eb     : ea;
    wire [10:0] my = a_bigger_mag ? mb     : ma;

    // ---- Align exponents (shift smaller significand right) ----
    wire [5:0]  ediff = (ex >= ey) ? (ex - ey) : 6'd0;

    // Limit the shift to 13 to avoid oversized shifts; anything larger underflows to ~0.
    wire [10:0] my_shifted =
        (ediff >= 6'd13) ? 11'd0 : (my >> ediff[3:0]); // truncate (no sticky for simplicity)

    // ---- Add/Sub significands based on signs ----
    // Use one extra bit to catch carry on addition: 12-bit workspace.
    wire same_sign = (sx == sy);

    wire [11:0] add_path = {1'b0, mx} + {1'b0, my_shifted};
    wire [11:0] sub_path = {1'b0, mx} - {1'b0, my_shifted};

    reg  [11:0] mant_pre;
    reg  [5:0]  exp_pre;     // keep a little headroom
    reg         sign_pre;

    always @* begin
        if (same_sign) begin
            // Addition
            mant_pre = add_path;
            exp_pre  = {1'b0, ex};
            sign_pre = sx;
        end else begin
            // Subtraction (mx >= my_shifted by construction)
            mant_pre = sub_path;
            exp_pre  = {1'b0, ex};
            sign_pre = sx; // sign of larger magnitude
        end
    end

    // ---- Normalize ----
    // Case 1: addition overflow (carry into bit 11) -> shift right, increment exponent
    wire add_overflow = same_sign && mant_pre[11];

    reg  [10:0] mant_norm; // 11-bit (1.hidden + frac)
    reg  [5:0]  exp_norm;

    // Leading-zero count for 11-bit value (positions 10..0)
    function [3:0] lzc11;
        input [10:0] x;
        integer i;
        reg found;
        begin
            lzc11 = 11;
            found = 0;
            for (i = 10; i >= 0; i = i - 1) begin
                if (!found && x[i]) begin
                    lzc11 = 10 - i;
                    found = 1;
                end
            end
            if (lzc11 > 11) lzc11 = 11;
        end
    endfunction

    reg [3:0] lz;
    reg [5:0] exp_tmp;
    reg [10:0] mant_tmp;

    always @* begin
        if (add_overflow) begin
            // Right shift by 1, increment exponent
            mant_norm = mant_pre[11:1];          // drop LSB (truncate)
            exp_norm  = exp_pre + 6'd1;
        end else begin
            // No immediate overflow
            // If result is zero, short-circuit
            if (mant_pre[10:0] == 11'd0) begin
                mant_norm = 11'd0;
                exp_norm  = 6'd0;
            end else begin
                // For subtraction, we may need to left-normalize
                if (!same_sign) begin
                    // Count leading zeros, shift left
                    // Target is to make bit10 = 1 for normal numbers
                    // If exponent can't support the shift, we drop to subnormal.
                    lz      = lzc11(mant_pre[10:0]);
                    exp_tmp = exp_pre;
                    mant_tmp= mant_pre[10:0];

                    if (lz != 0) begin
                        if (exp_tmp > lz) begin
                            mant_tmp = mant_tmp << lz;
                            exp_tmp  = exp_tmp - lz;
                        end else begin
                            // Subnormal: shift as much as exponent allows to reach exp=0
                            mant_tmp = mant_tmp << (exp_tmp[3:0]);
                            exp_tmp  = 6'd0;
                        end
                    end

                    mant_norm = mant_tmp;
                    exp_norm  = exp_tmp;
                end else begin
                    // Addition without overflow: already normalized if msb at bit10==1 (normal)
                    mant_norm = mant_pre[10:0];
                    exp_norm  = exp_pre;
                end
            end
        end
    end

    // ---- Pack result ----
    wire is_zero_res = (mant_norm == 11'd0);

    // Handle potential exponent overflow/underflow conservatively (clamp to max/min normal).
    wire [4:0] exp_clamped =
        (exp_norm[5] ? 5'd0 :
        (exp_norm > 6'd30) ? 5'd30 : exp_norm[4:0]);

    wire is_normal = (exp_clamped != 5'd0);

    wire [9:0] frac_out = is_normal ? mant_norm[9:0] : mant_norm[9:0];

    // Use effective sign of b for zero-sign choice as well
    wire       sign_out = is_zero_res ? (sa & sb_eff) : sign_pre;

    assign sum = is_zero_res
               ? 16'h0000
               : { sign_out, exp_clamped[4:0], frac_out };

endmodule
