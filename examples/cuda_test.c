#include <stdio.h>
#include "../src/nn_cuda.h"

int main(void) {
    NN_CUDA_ctx ctx = {0};
    nn_cuda_init(&ctx);

    float *p = NULL;
    nn_cuda_malloc(10, &p);
    if (p == NULL) {
        fprintf(stderr, "[ERROR]: Out of memory");
    } else {
        printf("p = %p\n", (void*)p);
    }
    nn_cuda_free(p);

    nn_cuda_destroy(&ctx);
    return 0;
}
