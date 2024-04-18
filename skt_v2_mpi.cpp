#include <hip/hip_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>

constexpr uint64_t TOTAL_ELEMENTS = 17179869184;
constexpr int STREAM_NUM = 128;
constexpr int THREADS_PER_PROCESS = 32;
constexpr int GPU_NUM = 8;
constexpr uint64_t SIZE = TOTAL_ELEMENTS / (THREADS_PER_PROCESS * GPU_NUM);

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

        void SetStartInput(float* input, int mpiRank, int threadIdx) {
            auto chunkSize =
                TOTAL_ELEMENTS /
                (GPU_NUM *
                 THREADS_PER_PROCESS); // Assume GPU_NUM same as the MPI_RANKS
            auto startIdx = mpiRank * THREADS_PER_PROCESS * chunkSize +
                            threadIdx * chunkSize;
            mInput = input + startIdx;
        }

        void Test(uint32_t deviceID) {
            hipSetDevice(deviceID);

            hipStream_t stream[STREAM_NUM];

            for (size_t i = 0; i < STREAM_NUM; i++) {
                hipStreamCreate(&stream[i]);
            }

            size_t size = mTotalThreads;
            // value
            std::vector<float> output(size, 2.0f); // Initialize with some value

            float *a, *b;

            hipMalloc((void**)&a, size * sizeof(float));

            hipMalloc((void**)&b, size * sizeof(float));

            for (size_t i = 0; i < STREAM_NUM; i++) {
                auto offset = i * size / STREAM_NUM;
#pragma omp barrier
                hipMemcpyAsync(a + offset, mInput + offset,
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
                cpu_output[i] = mInput[i] * 2.0f + 1.0f;
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
        float* mInput = nullptr;
};

int main(int argc, char** argv) {
    uint32_t totalThreads = SIZE;
    uint32_t threadsPerBlock = 1024;

    float* sharedMemBuf = nullptr;
    float* localSharedMemBuf = nullptr;

    MPI_Init(&argc, &argv);
    int totalProcesses, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    MPI_Comm nodecomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, processRank,
                        MPI_INFO_NULL, &nodecomm);

    int nodesize, noderank;
    MPI_Comm_size(nodecomm, &nodesize);
    MPI_Comm_rank(nodecomm, &noderank);

    uint64_t localsize = 0;
    if (0 == noderank) {
        localsize = TOTAL_ELEMENTS;
    }

    MPI_Win wintable;

    MPI_Win_allocate_shared(localsize * sizeof(float), sizeof(float),
                            MPI_INFO_NULL, nodecomm, &localSharedMemBuf,
                            &wintable);

    int* model;
    int flag;
    MPI_Win_get_attr(wintable, MPI_WIN_MODEL, &model, &flag);

    if (1 != flag) {
        printf("Attribute MPI_WIN_MODEL not defined\n");
        MPI_Finalize();
        return 1;
    } else {
        if (MPI_WIN_UNIFIED == *model) {
            if (processRank == 0) printf("Memory model is MPI_WIN_UNIFIED\n");
        } else {
            if (processRank == 0)
                printf("Memory model is *not* MPI_WIN_UNIFIED\n");

            MPI_Finalize();
            return 1;
        }
    }

    int windisp;
    MPI_Aint winsize;
    sharedMemBuf = localSharedMemBuf;
    if (noderank != 0) {
        MPI_Win_shared_query(wintable, 0, &winsize, &windisp, &sharedMemBuf);
    }

    MPI_Win_fence(0, wintable);

    if (0 == noderank) {
        std::cout << "TOTAL ELEMENT: " << TOTAL_ELEMENTS << std::endl;
        std::cout << "total input size (GB) = "
                  << TOTAL_ELEMENTS * sizeof(float) / 1024 / 1024 / 1024. << std::endl;
        for (size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
            sharedMemBuf[i] = 1.0f;
        }
    }

    MPI_Win_fence(0, wintable);

    TestHipKernelFmaAsyncCopy test(totalThreads, threadsPerBlock);

    omp_set_num_threads(THREADS_PER_PROCESS);

#pragma omp parallel
    {
        auto deviceID = processRank;
        auto threadIdx = omp_get_thread_num();
        printf("MPI rank = %d, GPU id = %d, thrad id = %d\n", processRank,
               deviceID, threadIdx);
        test.SetStartInput(sharedMemBuf, processRank, threadIdx);
        test.Test(deviceID);
    }

    MPI_Finalize();
    return 0;
}
