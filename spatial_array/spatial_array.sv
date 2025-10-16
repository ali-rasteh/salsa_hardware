// 8x8 spatial array of sa_cell PEs.
// Horizontal:  left_in  -> right_out  (west -> east)
// Vertical:    data_*   and result_*  (north -> south)

`include "sa_cell.v"
`include "accumulator.v"

`timescale 1ns / 1ns


module spatial_array_8x8 #(
    parameter DATA_WIDTH            = 16,
    parameter CTRL_WIDTH            = 9,
    parameter WD_BUFFER_DEPTH       = 16,
    parameter INPUT_BUFFER_DEPTH    = 4,
    parameter PARTIALS_BUFFER_DEPTH = 2
)(
    input  logic clk,
    input  logic rst,

    // -------- Per-PE control (driven externally) --------
    input  logic [CTRL_WIDTH-1:0] ctrl [0:7][0:7],
    input  logic [$clog2(WD_BUFFER_DEPTH)-1:0]        wd_buffer_pop_index       [0:7][0:7],
    input  logic [$clog2(INPUT_BUFFER_DEPTH)-1:0]     input_buffer_pop_index    [0:7][0:7],
    input  logic [$clog2(PARTIALS_BUFFER_DEPTH)-1:0]  partials_buffer_pop_index [0:7][0:7],
    input  logic                                      add_sub                   [0:7][0:7],
    input  logic                                      rst_accumulator           [0:7], // Per-column accumulator reset
    input  logic                                      enable_accumulator        [0:7], // Per-column accumulator enable

    // -------- Array boundary inputs --------
    // West edge (feeds left_in of column 0, each row has its own lane)
    input  logic [DATA_WIDTH-1:0] west_left_in [0:7],
    // North edge (feeds row 0 for each column)
    input  logic [DATA_WIDTH-1:0] north_data_in   [0:7],
    input  logic                  north_data_in_valid [0:7],
    input  logic [DATA_WIDTH-1:0] north_result_in [0:7],

    // -------- Array boundary outputs --------
    // East edge (right_out of column 7, per row)
    output logic [DATA_WIDTH-1:0] east_right_out [0:7],
    // South edge (from bottom row, per column)
    output logic [DATA_WIDTH-1:0] south_data_out     [0:7],
    output logic                  south_data_out_valid [0:7],
    output logic [DATA_WIDTH-1:0] south_result_out   [0:7]
);

    // ----------------------------------------------------------------
    // Interconnect wires
    // left_to_right[r][c] is the signal entering PE at row r, col c from the west.
    // We size columns as 0..8 so index 8 holds the east boundary (right_out of last PE).
    // Similarly, vertical busses have rows 0..8 where index 8 is the south boundary.
    // ----------------------------------------------------------------
    logic [DATA_WIDTH-1:0] h_left_to_right [0:7][0:8]; // horizontal data (left->right)
    logic [DATA_WIDTH-1:0] v_data          [0:8][0:7]; // vertical data (top->bottom)
    logic                  v_data_valid    [0:8][0:7]; // vertical valid
    logic [DATA_WIDTH-1:0] v_result        [0:8][0:7]; // vertical partial/result

    // ---------------- Boundary assignments ----------------
    // West boundary feeds column 0
    genvar rr, cc;
    generate
        for (rr = 0; rr < 8; rr++) begin : WEST_BOUNDARY_FEEDS
            assign h_left_to_right[rr][0] = west_left_in[rr];
        end
        // North boundary feeds row 0
        for (cc = 0; cc < 8; cc++) begin : NORTH_BOUNDARY_FEEDS
            assign v_data[0][cc]       = north_data_in[cc];
            assign v_data_valid[0][cc] = north_data_in_valid[cc];
            assign v_result[0][cc]     = north_result_in[cc];
        end
    endgenerate

    // ---------------- Grid instantiation ----------------
    generate
        for (rr = 0; rr < 8; rr++) begin : ROW
            for (cc = 0; cc < 8; cc++) begin : COL
                // Wires for this PE's outputs (to avoid accidental multiple drivers)
                logic [DATA_WIDTH-1:0] right_out_w;
                logic [DATA_WIDTH-1:0] result_out_w;
                logic [DATA_WIDTH-1:0] data_out_w;
                logic                  data_out_valid_w;

                sa_cell #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .CTRL_WIDTH(CTRL_WIDTH),
                    .WD_BUFFER_DEPTH(WD_BUFFER_DEPTH),
                    .INPUT_BUFFER_DEPTH(INPUT_BUFFER_DEPTH),
                    .PARTIALS_BUFFER_DEPTH(PARTIALS_BUFFER_DEPTH)
                ) u_pe (
                    .clk(clk),
                    .rst(rst),

                    .ctrl(ctrl[rr][cc]),
                    .left_in(h_left_to_right[rr][cc]),
                    .result_in(v_result[rr][cc]),
                    .data_in(v_data[rr][cc]),
                    .data_in_valid(v_data_valid[rr][cc]),

                    .wd_buffer_pop_index(wd_buffer_pop_index[rr][cc]),
                    .input_buffer_pop_index(input_buffer_pop_index[rr][cc]),
                    .partials_buffer_pop_index(partials_buffer_pop_index[rr][cc]),
                    .add_sub(add_sub[rr][cc]),

                    .right_out(right_out_w),
                    .result_out(result_out_w),
                    .data_out(data_out_w),
                    .data_out_valid(data_out_valid_w)
                );

                // Hook up to neighbors/boundaries
                // Eastward connection
                assign h_left_to_right[rr][cc+1] = right_out_w;
                // Southward connections
                assign v_data[rr+1][cc]       = data_out_w;
                assign v_data_valid[rr+1][cc] = data_out_valid_w;
                assign v_result[rr+1][cc]     = result_out_w;
            end
        end
    endgenerate

    // ---------------- Boundary outputs ----------------
    // East edge collects right_out of last column
    generate
        for (rr = 0; rr < 8; rr++) begin : EAST_BOUNDARY
            assign east_right_out[rr] = h_left_to_right[rr][8];
        end

        // South edge collects bottom row outputs
        for (cc = 0; cc < 8; cc++) begin : SOUTH_BOUNDARY
            // Pass-through result and valid from the southern edge of the grid
            assign south_result_out[cc]       = v_result[8][cc];
            assign south_data_out_valid[cc] = v_data_valid[8][cc];

            // Accumulate the southern-most (bottom row) data per column before exposing it
            accumulator_IEEE754_16bit #(
                .WIDTH(DATA_WIDTH)
            ) u_accum_south (
                .clk    (clk),
                .reset  (rst_accumulator[cc]),            // Active-high reset
                .enable (enable_accumulator[cc]),         // Accumulate only when data is valid
                .in     (v_data[8][cc]),     // Input from the bottom of the column
                .sum_out(south_data_out[cc]) // Output drives the south_data_out
            );
        end

    endgenerate




endmodule
