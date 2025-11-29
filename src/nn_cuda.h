#ifndef NEURAL_NETWORK_CUDA_H
#define NEURAL_NETWORK_CUDA_H

#include <stdbool.h>
#include <cublas_v2.h>

typedef struct NN_CUDA_ctx NN_CUDA_ctx;

#ifdef __cplusplus
extern "C" {
#endif

NN_CUDA_ctx *nn_cuda_init();

void nn_cuda_destroy(NN_CUDA_ctx *ctx);

void nn_cuda_malloc(size_t size, float **d);

void nn_cuda_free(float *d);

void nn_cuda_memset(float *d, int value, size_t count);

/* TODO: better parametrize with an enum */
void nn_cuda_memcpy_host_to_device(const NN_CUDA_ctx *ctx, float *dest, const float *src, size_t n);
void nn_cuda_memcpy_device_to_host(const NN_CUDA_ctx *ctx, float *dest, const float *src, size_t n);
void nn_cuda_memcpy_device_to_device(const NN_CUDA_ctx *ctx, float *dest, const float *src, size_t n);

void nn_cuda_matmul(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C);

void nn_cuda_matmul_t(const NN_CUDA_ctx *ctx, const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *C);

/*
*  'res', 'x' and 'y' are vectors of length 'len' allocated on the gpu.
*  This function computes res(i) = x(i) + alpha*y(i).
*/
void nn_cuda_add(const NN_CUDA_ctx *ctx, float *res, float *x, const float *y, size_t len, float alpha);

void nn_cuda_mult_elementwise(const NN_CUDA_ctx *ctx, float *x, float *y, size_t len);

void nn_cuda_sigmoid_vec(float *x, size_t len);
void nn_cuda_relu_vec(float *x, size_t len);
void nn_cuda_tanh_vec(float *x, size_t len);
void nn_cuda_softmax(float *x, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_NETWORK_CUDA_H */
