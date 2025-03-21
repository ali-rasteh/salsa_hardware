#include <hls_stream.h>
#include <iostream>
using namespace std;

#define N 1024              // Input size
#define KERNEL_SIZE 8     // Kernel size

// Top-level function for HLS synthesis
// The following directive instructs the HLS tool to limit the number of multiplier units used in operations.
void conv1D(const int input[N], const int kernel[KERNEL_SIZE], int output[N - KERNEL_SIZE + 1]) {
    #pragma HLS ALLOCATION instances=mul limit=64 operation  // Limit multipliers to at most 2

    // Loop over each valid output index
    // #pragma HLS PIPELINE off
    for (int i = 0; i < N - KERNEL_SIZE + 1; i++) {
        int sum = 0;
        // Convolve the kernel with the input data
        for (int j = 0; j < KERNEL_SIZE; j++) {
            // #pragma HLS UNROLL factor=1
            sum += input[i+j] * kernel[j];
        }
        output[i] = sum;
    }
}
