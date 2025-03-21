#include <hls_stream.h>
#include <ap_int.h>

#define KERNEL_SIZE 8

// Define the data type for stream elements.
// Adjust the bit-width (here 16) to match the M bits per second requirement.
typedef ap_int<16> data_t;

// Top-level function for HLS synthesis using streaming interfaces
void conv1D_stream(
    hls::stream<data_t> &in_stream,      // input stream of data_t elements
    const data_t kernel[KERNEL_SIZE],    // convolution kernel (constant coefficients)
    hls::stream<data_t> &out_stream,       // output stream of convolution results
    int num_samples                      // total number of samples to process (must be >= KERNEL_SIZE)
) {
    // Specify streaming interfaces (AXI-Stream) and control interface (AXI-Lite) pragmas
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=kernel bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=num_samples bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // Local buffer to hold the sliding window of input samples.
    data_t window[KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=window complete

    // Preload the sliding window with the first (KERNEL_SIZE-1) samples.
    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
        #pragma HLS PIPELINE II=1
        window[i] = in_stream.read();
    }

    // Process the incoming stream sample by sample.
    // The loop will process num_samples - (KERNEL_SIZE - 1) outputs.
    for (int i = 0; i < num_samples - KERNEL_SIZE + 1; i++) {
        #pragma HLS PIPELINE II=1

        // Read new sample and update the window.
        window[KERNEL_SIZE - 1] = in_stream.read();

        // Compute the convolution sum for the current window.
        data_t sum = 0;
        for (int j = 0; j < KERNEL_SIZE; j++) {
            #pragma HLS UNROLL
            sum += window[j] * kernel[j];
        }

        // Write the computed result to the output stream.
        out_stream.write(sum);

        // Shift the window: move each element one position to the left.
        for (int k = 0; k < KERNEL_SIZE - 1; k++) {
            #pragma HLS UNROLL
            window[k] = window[k + 1];
        }
    }
}
