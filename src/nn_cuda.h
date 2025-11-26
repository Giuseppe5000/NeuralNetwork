#ifndef NEURAL_NETWORK_CUDA
#define NEURAL_NETWORK_CUDA

#include <stdbool.h>
#include <cublas_v2.h>

/* TODO: maybe it is a good idea to make an opaque type */
typedef struct {
    cublasHandle_t cublasH;
    cudaStream_t stream;
} NN_CUDA_ctx;

#ifdef __cplusplus
extern "C" {
#endif

void nn_cuda_init(NN_CUDA_ctx *ctx);

void nn_cuda_destroy(NN_CUDA_ctx *ctx);

void nn_cuda_alloc(size_t size, float **d);

void nn_cuda_free(float *d);

void nn_cuda_matmul(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C, bool transpose_B);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_NETWORK_CUDA */
