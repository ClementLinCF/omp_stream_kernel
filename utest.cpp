#include <hip/hip_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

using namespace std;

__global__ void fmaKernel(float* input, float* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        output[tid] = input[tid] * 2.0f + 1.0f; // Fused multiply-add operation
    }
}

class TestHipKernelFmaAsyncCopy {
    public:
        TestHipKernelFmaAsyncCopy() = default;
        TestHipKernelFmaAsyncCopy(uint32_t totalThreads,
                                  uint32_t threadsPerBlock)
            : mTotalThreads(totalThreads), mThreadsPerBlock(threadsPerBlock) {
            mTotalBlocks =
                (mTotalThreads + mThreadsPerBlock - 1) / mThreadsPerBlock;
        }
        void Test(hipStream_t stream) {
            // Define host data (replace with your actual data)
            size_t size = mTotalThreads;
            vector<float> input(size, 1.0f);  // Initialize with some value
            vector<float> output(size, 2.0f); // Initialize with some value

            float *a, *b;
            hipMalloc((void**)&a, size * sizeof(float));
            hipMalloc((void**)&b, size * sizeof(float));

            hipMemcpyAsync(a, input.data(), size * sizeof(float),
                           hipMemcpyHostToDevice, stream);

            hipLaunchKernelGGL(fmaKernel, dim3(mTotalBlocks, 1, 1),
                               dim3(mThreadsPerBlock, 1, 1), 0, stream, a, b,
                               size);

            hipMemcpyAsync(output.data(), b, size * sizeof(float),
                           hipMemcpyDeviceToHost, stream);


            hipStreamSynchronize(stream);

            vector<float> cpu_output(size);
            // FMA on CPU
            for (int i = 0; i < size; i++) {
                cpu_output[i] = input[i] * 2.0f + 1.0f;
            }

            bool success = true;
            for (int i = 0; i < size; i++) {
                if (output[i] != cpu_output[i]) {
                    success = false;
                    std::cout << "Error: output[" << i << "] != cpu_output["
                              << i << "]" << std::endl;
                    std::cout << output[i] << std::endl;
                    std::cout << cpu_output[i] << std::endl;
                    break;
                }
            }
            if (success) {
                std::cout << "Success!" << std::endl;
            }

            hipFree(a);
            hipFree(b);
        }

    private:
        uint32_t mTotalThreads = 0;
        uint32_t mThreadsPerBlock = 0;
        uint32_t mTotalBlocks = 0;
};

int main() {
    uint32_t totalThreads = 1024 * 1024;
    uint32_t threadsPerBlock = 1024;
    TestHipKernelFmaAsyncCopy test(totalThreads, threadsPerBlock);

    hipStream_t sharedStream;
    hipStreamCreate(&sharedStream);

#pragma omp parallel num_threads(8)
    { test.Test(sharedStream); }
    hipStreamDestroy(sharedStream);
    return 0;
}
