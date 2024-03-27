#include <hip/hip_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

#define STREAM_NUM 32

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
        void Test(uint32_t deviceID) {
            hipSetDevice(deviceID);

            hipStream_t stream[STREAM_NUM];

            for (size_t i = 0; i < STREAM_NUM; i++) {
                hipStreamCreate(&stream[i]);
            }

            size_t size = mTotalThreads;
            std::vector<float> input(size, 1.0f);  // Initialize with some value
            std::vector<float> output(size, 2.0f); // Initialize with some value

            float *a, *b;

            hipMalloc((void**)&a, size * sizeof(float));

            hipMalloc((void**)&b, size * sizeof(float));

            for (size_t i = 0; i < STREAM_NUM; i++) {
                auto offset = i * size / STREAM_NUM;

                hipMemcpyAsync(a + offset, input.data() + offset,
                               size * sizeof(float) / STREAM_NUM,
                               hipMemcpyHostToDevice, stream[i]);

                hipLaunchKernelGGL(fmaKernel, dim3(mTotalBlocks, 1, 1),
                                   dim3(mThreadsPerBlock, 1, 1), 0, stream[i],
                                   a + offset, b + offset, size / STREAM_NUM);

                hipMemcpyAsync(output.data() + offset, b + offset,
                               size * sizeof(float) / STREAM_NUM,
                               hipMemcpyDeviceToHost, stream[i]);
            }

            for (size_t i = 0; i < STREAM_NUM; i++) {
                hipStreamSynchronize(stream[i]);
            }

            std::vector<float> cpu_output(size);
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
            for (size_t i = 0; i < STREAM_NUM; i++) {
                hipStreamDestroy(stream[i]);
            }
        }

    private:
        uint32_t mTotalThreads = 0;
        uint32_t mThreadsPerBlock = 0;
        uint32_t mTotalBlocks = 0;
};

int main() {
    uint32_t totalThreads = 1024 * 1024 * 1024;
    uint32_t threadsPerBlock = 1024;
    uint32_t totalGpuNum = 8;

    TestHipKernelFmaAsyncCopy test(totalThreads, threadsPerBlock);

    omp_set_num_threads(totalGpuNum);

#pragma omp parallel
    {
        auto deviceID = omp_get_thread_num();
        test.Test(deviceID);
    }

    return 0;
}
