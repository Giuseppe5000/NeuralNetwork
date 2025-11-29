/* Reference: https://github.com/NVIDIA/CUDALibrarySamples/blob/main/cuBLAS/Level-3/gemm/cublas_gemm_example.cu */

#include "nn_cuda.h"
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

struct NN_CUDA_ctx {
    cublasHandle_t cublasH; /* TODO: rename to cublasHandle */
    cudaStream_t stream;
};

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
    NN_CUDA_ctx *nn_cuda_init() {
        NN_CUDA_ctx *ctx = (NN_CUDA_ctx *) malloc(sizeof(NN_CUDA_ctx));

        /* Create cublas handle, bind a stream */
        CUBLAS_CHECK(cublasCreate(&ctx->cublasH));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
        CUBLAS_CHECK(cublasSetStream(ctx->cublasH, ctx->stream));

        return ctx;
    }

    void nn_cuda_destroy(NN_CUDA_ctx *ctx) {
        CUBLAS_CHECK(cublasDestroy(ctx->cublasH));
        CUDA_CHECK(cudaStreamDestroy(ctx->stream));
        CUDA_CHECK(cudaDeviceReset());
        free(ctx);
    }

    void nn_cuda_malloc(size_t size, float **d) {
        CUDA_CHECK(cudaMalloc(d, size));
    }

    void nn_cuda_free(float *d) {
        CUDA_CHECK(cudaFree(d));
    }

    void nn_cuda_memset(float *d, int value, size_t count) {
        CUDA_CHECK(cudaMemset(d, value, count));
    }

    void nn_cuda_memcpy_to_device(const NN_CUDA_ctx *ctx, float *dest, const float *src, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    void nn_cuda_memcpy_to_host(const NN_CUDA_ctx *ctx, float *dest, const float *src, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    static void _nn_cuda_matmul(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C, bool transpose_B) {
        size_t C_rows = A_rows;

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

        /* Compute */
        CUBLAS_CHECK(cublasSgemm(ctx->cublasH, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));

        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    void nn_cuda_matmul(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C) {
        _nn_cuda_matmul(ctx, A, A_rows, A_cols, B, B_rows, B_cols, C, false);
    }

    void nn_cuda_matmul_t(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C) {
        _nn_cuda_matmul(ctx, A, A_rows, A_cols, B, B_rows, B_cols, C, true);
    }

    void nn_cuda_add(const NN_CUDA_ctx *ctx, float *x,  const float *y, size_t len, float alpha) {
        const int incx = 1;
        const int incy = 1;

        CUBLAS_CHECK(cublasSaxpy(ctx->cublasH, len, &alpha, y, incx, x, incy));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }

    /* ============== Activation functions and derivative ============== */
    __global__ void sigmoidKernel(float *x, size_t len) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        x[i] = 1.0 / (1.0 + expf(-x[i]));
    }

    void nn_cuda_sigmoid_vec(float *x, size_t len) {
        const int blocksize = 512;
        int nblocks = (int)((len + blocksize - 1) / blocksize);
        sigmoidKernel<<<nblocks, blocksize>>>(x, len);
    }

    void nn_cuda_relu_vec(float *x, size_t len) {
        assert(0 && "Not implemented");
    }
    void nn_cuda_tanh_vec(float *x, size_t len) {
        assert(0 && "Not implemented");
    }

    // __global__ void softmaxKernel(float *x, size_t len) {
    //     int i = threadIdx.x + blockIdx.x * blockDim.x;
    //     x[i] = 1.0 / (1.0 + expf(-x[i]));
    // }

    void nn_cuda_softmax(float *x, size_t len) {
        assert(0 && "Not implemented");

        /* asum and amax can be useful */

        // const int blocksize = 512;
        // int nblocks = (int)((len + blocksize - 1) / blocksize);
        // softmaxKernel<<<nblocks, blocksize>>>(x, len);
    }

    /* ================================================================= */

}
