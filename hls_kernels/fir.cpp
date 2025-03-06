#include "hls_stream.h"
#define N_TAPS 16



// FIR filter function using Vitis HLS
void fir_filter(hls::stream<float> &in_stream, hls::stream<float> &out_stream, const float coeff[N_TAPS]) {
    // Interface pragmas: use AXIS for streaming and no control protocol for a pure dataflow design
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Static shift register for the delay line
    static float shift_reg[N_TAPS];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete

    // Read a new input sample
    float x = in_stream.read();

    // Shift register update: move previous samples down the delay line
    SHIFT_LOOP: for (int i = N_TAPS - 1; i > 0; i--) {
        #pragma HLS PIPELINE II=1
        shift_reg[i] = shift_reg[i - 1];
    }
    shift_reg[0] = x;

    // Multiply–Accumulate (MAC) operation for FIR filtering
    float acc = 0;
    MAC_LOOP: for (int i = 0; i < N_TAPS; i++) {
        #pragma HLS PIPELINE II=1
        // Constrain the multiplication operation to at most 2 instances,
        // effectively limiting the number of MAC units in the hardware.
        #pragma HLS ALLOCATION instances=mul limit=2 operation
        acc += shift_reg[i] * coeff[i];
    }

    // Write the filtered output
    out_stream.write(acc);
}


