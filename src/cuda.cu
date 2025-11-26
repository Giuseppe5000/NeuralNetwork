/* Reference: https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLAS/Level-3/gemm/cublas_gemm_example.cu */

#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* CUDA API error checking */
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* cuBLAS API error checking */
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

extern "C" {
    void cuda_init(cublasHandle_t cublasH, cudaStream_t stream) {
        /* Create cublas handle, bind a stream */
        CUBLAS_CHECK(cublasCreate(&cublasH));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    }

    void cuda_destroy(cublasHandle_t cublasH, cudaStream_t stream) {
        CUBLAS_CHECK(cublasDestroy(cublasH));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaDeviceReset());
    }

    void cuda_alloc(size_t size, float *d) {
        CUDA_CHECK(cudaMalloc((void **)(&d), sizeof(float) * size));
    }

    void cuda_free(float *d) {
        CUDA_CHECK(cudaFree(d));
    }

    void cuda_matmul(cublasHandle_t cublasH, cudaStream_t stream, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C, bool transpose_B) {
        size_t C_rows = A_rows;
        // size_t C_cols = B_cols;
        // if (transpose_B) {
        //     C_cols = B_rows;
        // }

        const int m = A_rows;
        int n = B_cols;
        const int k = A_cols;
        const int lda = A_rows;
        const int ldb = B_rows;
        const int ldc = C_rows;
        if (transpose_B) {
            n = B_rows;
        }

        const float alpha = 1.0;
        const float beta = 0.0;

        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        if (transpose_B) {
            transb = CUBLAS_OP_T;
        }

        /* Copy data to device */
        // CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeof(float) * A_rows * A_cols, cudaMemcpyHostToDevice, stream));
        // CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeof(float) * B_rows * B_cols, cudaMemcpyHostToDevice, stream));

        /* Compute */
        CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));

        /* Copy data to host */
        // CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeof(float) * C_rows * C_cols, cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
