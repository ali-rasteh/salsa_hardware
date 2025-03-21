#include <iostream>
#include <hls_stream.h>
#include <ap_int.h>

#define KERNEL_SIZE 8
#define NUM_SAMPLES 1024

// Define the data type for stream elements (must match conv1D_stream definition).
typedef ap_int<16> data_t;

// Declaration of the conv1D_stream function.
void conv1D_stream(
    hls::stream<data_t> &in_stream,      
    const data_t kernel[KERNEL_SIZE],    
    hls::stream<data_t> &out_stream,       
    int num_samples                      
);

int main() {
    // Create input and output streams.
    hls::stream<data_t> input_stream;
    hls::stream<data_t> output_stream;

    // Define a test kernel. (Example: alternating positive and negative coefficients)
    data_t kernel[KERNEL_SIZE] = {1, -1, 2, -2, 3, -3, 4, -4};

    // Initialize the input stream with NUM_SAMPLES values.
    // For example, fill with increasing integers.
    for (int i = 0; i < NUM_SAMPLES; i++) {
        input_stream.write(i);
    }

    // Call the streaming convolution function.
    // This processes NUM_SAMPLES samples and produces (NUM_SAMPLES - KERNEL_SIZE + 1) output samples.
    conv1D_stream(input_stream, kernel, output_stream, NUM_SAMPLES);

    // Number of output samples to expect.
    int num_outputs = NUM_SAMPLES - KERNEL_SIZE + 1;
    std::cout << "Convolution Output:" << std::endl;

    // Read and print the results from the output stream.
    for (int i = 0; i < num_outputs; i++) {
        data_t result = output_stream.read();
        std::cout << "Output[" << i << "] = " << result << std::endl;
    }

    return 0;
}
