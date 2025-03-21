#include <iostream>
using namespace std;

#define N 1024
#define KERNEL_SIZE 8

// Declaration of the top-level function
void conv1D(const int input[N], const int kernel[KERNEL_SIZE], int output[N - KERNEL_SIZE + 1]);

int main() {
    // Example input data and kernel
    int input[N] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int kernel[KERNEL_SIZE] = {1, 0, -1}; // Simple edge-detection kernel (difference operator)
    int output[N - KERNEL_SIZE + 1];

    // Call the convolution function
    conv1D(input, kernel, output);

    // Display the output
    for (int i = 0; i < N - KERNEL_SIZE + 1; i++) {
        cout << "output[" << i << "] = " << output[i] << endl;
    }
    return 0;
}
