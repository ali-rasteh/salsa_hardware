#include <hls_stream.h>
#include <ap_int.h>


// extern "C" {


void conv1d(const float *in, float *out, const float *kernel, 
            int in_size, int kernel_size, int stride) {
#pragma HLS INTERFACE m_axi port=in      offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out     offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=kernel  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=in_size     bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_size bundle=control
#pragma HLS INTERFACE s_axilite port=stride      bundle=control
#pragma HLS INTERFACE s_axilite port=return      bundle=control

    // Compute the number of output elements
    int out_size = (in_size - kernel_size) / stride + 1;

    // Outer loop over output elements
    conv1d_loop: for (int i = 0; i < out_size; i++) {
    #pragma HLS PIPELINE II=1
        float acc = 0;
        // Inner loop: perform convolution (MAC operations)
        kernel_loop: for (int j = 0; j < kernel_size; j++) {
            // Limit the multiplier (MAC unit) to a single instance
            #pragma HLS ALLOCATION instances=mul limit=1 operation
            float product = in[i * stride + j] * kernel[j];
            acc += product;
        }
        out[i] = acc;
    }
}


// }

