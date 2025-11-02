#include "neural_network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <float.h>

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
    length = 'units_configuration_len' - 1.
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
static float glorot(size_t n_in, size_t n_out) {
    const float x = sqrtf(6.0 / (n_in + n_out));
    return randf(-x, x);
}

/*
'res' needs to be a matrix of 'A_rows' rows and 'B_cols' columns.
*/
static void nn_matrix_mul(const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *res) {
    if (A_cols != B_rows) {
        fprintf(stderr, "[ERROR]: Trying to mult A * B, but A_cols != B_rows\n");
        exit(1);
    }

    for (size_t row = 0; row < A_rows; ++row) {
        for (size_t col = 0; col < B_cols; ++col) {
            float dot_prod = 0.0;

            for (size_t i = 0; i < A_rows; ++i) {
                dot_prod += A[row*A_cols + i] * B[i*B_cols + col];
            }

            res[row*B_cols + col] = dot_prod;
        }
    }
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



NN *nn_init(size_t *units_configuration, size_t units_configuration_len, enum Activation *units_activation, enum Weight_initialization w_init) {
    NN *nn = nn_malloc(sizeof(NN));

    /* Copy units_configuration into the struct */
    nn->units_configuration = nn_malloc(units_configuration_len * sizeof(size_t));
    nn->units_configuration_len = units_configuration_len;
    memcpy(nn->units_configuration, units_configuration, units_configuration_len * sizeof(size_t));

    /* Weights and layers initialize */
    nn->weights_len = 0;

    for (size_t i = 0; i < units_configuration_len - 1; ++i) {
        /*
        (units_configuration[i] * units_configuration[i+1]) = N x M matrix weights
        units_configuration[i+1] = weights for bias term
        */
        nn->weights_len += units_configuration[i] * units_configuration[i+1] + units_configuration[i+1];
    }
    nn->weights = nn_malloc(nn->weights_len * sizeof(float));

    nn->layers_len = units_configuration_len - 1;
    nn->layers = nn_malloc(nn->layers_len * sizeof(float *));
    size_t counter = 0;
    for (size_t i = 0; i < nn->layers_len; ++i) {
        nn->layers[i] = nn->weights + counter;
        counter += units_configuration[i] * units_configuration[i+1] + units_configuration[i+1];
    }

    srand(time(NULL));
    for(size_t i = 0; i < nn->layers_len; ++i) {
        const size_t layer_size = units_configuration[i] * units_configuration[i+1] + units_configuration[i+1];

        for (size_t j = 0; j < layer_size; ++j) {
            float *weight = nn->layers[i] + j;

            switch (w_init) {
                case NN_UNIFORM:
                    *weight = randf(-0.01,0.01);
                    break;
                case NN_GLOROT:
                    *weight = glorot(units_configuration[i] + 1, units_configuration[i+1]);
                    break;
                default:
                    assert(0 && "Unreachable");
            }
        }
    }

    /* Activation functions initialize */
    nn->activations = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation *));
    nn->activations_derivative = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation_derivative *));

    for (size_t i = 0; i < units_configuration_len - 1; ++i) {
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

void nn_free(NN *nn) {
    free(nn->units_configuration);
    free(nn->weights);
    free(nn->layers);
    free(nn->activations);
    free(nn->activations_derivative);
    free(nn);
}

void nn_predict(NN *nn, const float *x, float *out) {
    size_t x_cols = nn->units_configuration[0];
    const size_t x_rows = 1;

    /*
    Find the unit configuration with maximum neurons.
    In this way we are sure that 'input' can contain
    all the intermediate results during the forward.
    */
    size_t max_neurons = 0;

    for (size_t i = 0; i < nn->units_configuration_len; ++i) {
        if (nn->units_configuration[i] > max_neurons) {
            max_neurons = nn->units_configuration[i];
        }
    }

    float input[max_neurons];
    input[0] = 1.0; /* Bias */
    memcpy(input + 1, x, x_cols * sizeof(float));

    /* Feed forward through the nn layers */
    for (size_t i = 0; i < nn->layers_len; ++i) {
        size_t res_len = nn->units_configuration[i+1];
        float res[res_len];

        nn_matrix_mul(
            input, x_rows, x_cols + 1, /* + 1 is for multiply the bias weights */
            nn->layers[i], nn->units_configuration[i] + 1, nn->units_configuration[i+1],
            res
        );

        /* Applying activation function */
        for (size_t j = 0; j < res_len; ++j) {
            nn->activations[i](res[j]);
        }

        /* 'res' is the new input */
        memcpy(input + 1, res, res_len * sizeof(float));
        x_cols = nn->units_configuration[i+1];
    }

    memcpy(out, input, nn->units_configuration[nn->units_configuration_len - 1] * sizeof(float));
}

void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, float learning_rate, float err_threshold) {
    float error = FLT_MAX;

    while (error > err_threshold) {

        /* Getting the current error, using the Softmax */
        error = 0.0;
        const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
        float out[out_len];
        for (size_t i = 0; i < train_len; ++i) {
            nn_predict(nn, x_train + i*nn->units_configuration[0], out);

            for (size_t j = 0; j < out_len; ++j) {
                error += powf(y_train[i*out_len + j] - out[i*out_len + j], 2);
            }
        }
        error *= 1.0/(2.0*out_len*train_len);

        printf("Error = %f\n", error);
    }
}
