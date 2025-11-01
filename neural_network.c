#include "neural_network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

/* ======================== Data structures ======================== */
typedef float (*nn_activation)(float);
typedef float (*nn_activation_derivative)(float);

struct NN {
    /*
    NN architecture.
    */
    size_t *units_configuration;
    size_t units_configuration_len;

    /*
    'layers' is a different view of the 'weights'.
    In fact it is an array of pointer, and each pointer points to the start of a matrix in 'weights'.

    For example, layers[1] = pointer to the first element of second layer.

    This is convinient for moving between the nn layers.

    Obviously 'layer_len' = 'units_configuration_len' - 1.
    */
    float **layers;
    size_t layers_len;

    /*
    Simply all the weights in the nn.
    'weights_len' is equal to the sum of each subsequent pair of 'units_configuration'.

    Example:
        size_t units_configuration[] = {3, 2, 1};

        Then 'weights_len' = 3*2 + 2*1 = 6 + 2 = 8.
    */
    float *weights;
    size_t weights_len;

    /*
    Array of activation function (and its derivative).
    length = 'units_configuration_len'.
    */
    nn_activation *activations;
    nn_activation_derivative *activations_derivative;
};
/* ================================================================= */



/* ======================= Helper functions ======================== */

/*
Wrapping malloc with this helper function,
handling OOM with exit and logging the error.
*/
static void *nn_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "[ERROR]: Out of memory.");
        exit(1);
    }
    return ptr;
}

/*
Returns a random float between 'min' and 'max'.
*/
static float randf(float min, float max) {
    return min + (max - min) * ((float)rand()/(float)RAND_MAX);
}

/*
Glorot initialization
https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization
*/
static float glorot(size_t n_in) {
    const size_t n_out = 1;
    const float x = sqrtf(6.0 / (n_in + n_out));
    return randf(-x, x);
}
/* ================================================================= */



/* ============== Activation functions and derivative ============== */
static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float relu(float x) {
    return x > 0.0 ? x : 0.0;
}

static float sigmoid_derivative(float x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

static float relu_derivative(float x) {
    /* Not derivable when x = 0, so for practical reasons the value in x = 0 is 0 */
    return x > 0.0 ? 1.0 : 0.0;
}

static float tanh_derivative(float x) {
    return 1.0 - powf(tanhf(x), 2);
}
/* ================================================================= */



NN *NN_init(size_t *units_configuration, size_t units_configuration_len, enum Activation *units_activation, enum Weight_initialization w_init) {
    NN *nn = nn_malloc(sizeof(NN));

    /* Copy units_configuration into the struct */
    nn->units_configuration = nn_malloc(units_configuration_len * sizeof(size_t));
    memcpy(nn->units_configuration, units_configuration, units_configuration_len * sizeof(size_t));

    /* Weights initialize */
    nn->weights_len = 0;

    for (size_t i = 0; i < units_configuration_len - 1; ++i) {
        nn->weights_len += units_configuration[i] * units_configuration[i+1];
    }

    nn->weights = nn_malloc(nn->weights_len * sizeof(float));

    srand(time(NULL));
    for (size_t i = 0; i < nn->weights_len; ++i) {
        switch (w_init) {
            case NN_UNIFORM:
                nn->weights[i] = randf(-0.01,0.01);
                break;
            case NN_GLOROT:
                nn->weights[i] = glorot(nn->weights_len);
                break;
            default:
                assert(0 && "Unreachable");
        }
    }

    /* Layers initialize */
    nn->layers_len = units_configuration_len - 1;
    nn->layers = nn_malloc(nn->layers_len * sizeof(float *));

    size_t counter = 0;
    for (size_t i = 0; i < nn->layers_len; ++i) {
        nn->layers[i] = nn->weights + counter;
        counter += units_configuration[i] * units_configuration[i+1];
    }

    /* Activation functions initialize */
    nn->activations = nn_malloc(units_configuration_len * sizeof(nn_activation *));
    nn->activations_derivative = nn_malloc(units_configuration_len * sizeof(nn_activation_derivative *));

    for (size_t i = 0; i < units_configuration_len; ++i) {
        switch (units_activation[i]) {
            case NN_SIGMOID:
                (nn->activations)[i] = sigmoid;
                (nn->activations_derivative)[i] = sigmoid_derivative;
                break;
            case NN_RELU:
                (nn->activations)[i] = relu;
                (nn->activations_derivative)[i] = relu_derivative;
                break;
            case NN_TANH:
                (nn->activations)[i] = tanhf;
                (nn->activations_derivative)[i] = tanh_derivative;
                break;
            default:
                assert(0 && "Unreachable");
        }
    }

    return nn;
}
