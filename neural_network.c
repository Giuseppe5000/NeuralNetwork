#include "neural_network.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    /*
    Array for storing the intermediate activations of the feed forward (contains even input and output).

    Its length has to be the sum of the elements of nn->units_configuration + the biases:
    so, (nn->units_configuration[0] + 1) +  (nn->units_configuration[1] + 1) + .. + (nn->units_configuration[nn->units_configuration_len - 1]).
    */
    float *intermediate_activations;
    size_t intermediate_activations_len;
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
Glorot initialization.
https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization
*/
static float glorot(size_t fan_in, size_t fan_out) {
    const float x = sqrtf(6.0 / (fan_out + fan_in));
    return randf(-x, x);
}

/*
He initialization using Box-Muller transform.
https://en.wikipedia.org/wiki/Weight_initialization#He_initialization
*/
static float he(size_t fan_out) {
    const float stddev = sqrtf(2.0 / fan_out);
    const float u1 = (float)rand() / (float)RAND_MAX;
    const float u2 = (float)rand() / (float)RAND_MAX;
    const float z0 = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * M_PI * u2);
    return z0 * stddev;
}

/*
Multiply 'A' to 'B', putting the result in 'res'.
'res' needs to be a matrix of 'A_rows' rows and 'B_cols' columns.
*/
static void nn_matrix_mul(const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *res) {
    if (A_cols != B_rows) {
        fprintf(stderr, "[ERROR]: Trying to mult A * B, but A_cols != B_rows\n");
        exit(1);
    }

    memset(res, 0, A_rows*B_cols*sizeof(float));

    for (size_t row = 0; row < A_rows; ++row) {
        for (size_t col = 0; col < B_cols; ++col) {
            for (size_t i = 0; i < A_cols; ++i) {
                res[row*B_cols + col] += A[row*A_cols + i] * B[i*B_cols + col];
            }
        }
    }
}

/*
Multiply 'A' to the transpose of 'B' directly, putting the result in res.
'res' needs to be a matrix of 'A_rows' rows and 'B_rows' columns.
*/
static void nn_matrix_mul_t(const float *A, size_t A_rows, size_t A_cols, const float *B, size_t B_rows, size_t B_cols, float *res) {
    if (A_cols != B_cols) {
        fprintf(stderr, "[ERROR]: Trying to mult A * transpose(B), but A_cols != B_cols\n");
        exit(1);
    }

    memset(res, 0, A_rows*B_rows*sizeof(float));

    for (size_t row = 0; row < A_rows; ++row) {
        for (size_t col = 0; col < B_rows; ++col) {
            for (size_t i = 0; i < A_cols; ++i) {
                res[row*B_rows + col] += A[row*A_cols + i] * B[col*B_cols + i];
            }
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
                    *weight = glorot(units_configuration[i+1], units_configuration[i] + 1);
                    break;
                case NN_HE:
                    *weight = he(units_configuration[i+1]);
                    break;
                default:
                    assert(0 && "Unreachable");
            }
        }
    }

    /* Activation functions initialize */
    nn->activations = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation));
    nn->activations_derivative = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation_derivative));

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
                fprintf(stderr, "[ERROR]: Check if the 'units_activation' length == units_configuration_len - 1.\n");
                exit(1);
        }
    }

    /* Intermidiate activations initialize */
    nn->intermediate_activations_len = 0;

    for (size_t i = 0; i < nn->units_configuration_len; ++i) {
        nn->intermediate_activations_len += nn->units_configuration[i] + 1; /* Number of neurons + bias */
    }
    nn->intermediate_activations_len--; /* Last layer doesn't have bias */

    nn->intermediate_activations = nn_malloc(nn->intermediate_activations_len * sizeof(float));

    return nn;
}

void nn_free(NN *nn) {
    free(nn->units_configuration);
    free(nn->weights);
    free(nn->layers);
    free(nn->activations);
    free(nn->activations_derivative);
    free(nn->intermediate_activations);
    free(nn);
}

/*
Feed forward the NN.

'x' is an array of length 'nn->units_configuration[0]'.
The prediction result will be put in the 'res' array of length 'units_configuration[nn->units_configuration_len - 1]'.

The intermediate products will be put in the 'intermediate_products' array IF it is not NULL.
Its length has to be the sum of the elements of nn->units_configuration + the biases:
    so, (nn->units_configuration[0] + 1) +  (nn->units_configuration[1] + 1) + .. + (nn->units_configuration[nn->units_configuration_len - 1]).
*/
static void nn_feed_forward(NN *nn, const float *x, float *intermediate_products) {
    size_t x_cols = nn->units_configuration[0];
    const size_t x_rows = 1;

    /* Copy input into intermediate_activations */
    nn->intermediate_activations[0] = 1.0; /* Bias */
    memcpy(nn->intermediate_activations + 1, x, x_cols * sizeof(float));

    /* Copy input into intermidiate product */
    float *intermediate_i = NULL;
    if (intermediate_products != NULL) {
        intermediate_products[0] = 1.0;
        memcpy(intermediate_products + 1, x, x_cols * sizeof(float));
        intermediate_i = intermediate_products + x_cols + 1;
    }

    /* Feed forward through the nn layers */
    float *activations_i = nn->intermediate_activations; /* Activations of units i */
    float *activations_i_next = nn->intermediate_activations + x_cols + 1; /* Activations of units i+1 */

    for (size_t i = 0; i < nn->layers_len; ++i) {
        size_t res_len = nn->units_configuration[i+1];

        const unsigned int is_not_last_layer = (i != nn->layers_len - 1) ? 1 : 0;

        nn_matrix_mul(
            activations_i, x_rows, x_cols + 1, /* + 1 is for multiply the bias weights */
            nn->layers[i], nn->units_configuration[i] + 1, nn->units_configuration[i+1],
            activations_i_next + is_not_last_layer
        );

        /* Bias for next iteration */
        if (is_not_last_layer) {
            activations_i_next[0] = 1.0;
        }

        /* Saving intermediate products */
        if (intermediate_products != NULL) {
            if (is_not_last_layer) {
                memcpy(intermediate_i, activations_i_next, (res_len+1)*sizeof(float));
            }
            else {
                memcpy(intermediate_i, activations_i_next, (res_len)*sizeof(float));
            }
        }

        /* Applying activation function */
        for (size_t j = is_not_last_layer; j < res_len + is_not_last_layer; ++j) {
            activations_i_next[j] = nn->activations[i](activations_i_next[j]);
        }

        activations_i = activations_i_next;
        activations_i_next += res_len + is_not_last_layer;
        if (intermediate_products != NULL) intermediate_i += res_len + is_not_last_layer;
        x_cols = nn->units_configuration[i+1];
    }
}

void nn_predict(NN *nn, const float *x, float *out) {
    nn_feed_forward(nn, x, NULL);

    /* Copy the output of the last elements of intermediate activations in 'out' */
    size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    size_t out_index = nn->intermediate_activations_len - out_len;
    memcpy(out, nn->intermediate_activations + out_index, out_len*sizeof(float));
}

static void backpropagation(const NN *nn, size_t batch_i, const float *y_train, const float *intermediate_products, size_t intermediate_products_len, float *deltas, size_t deltas_len, float *gradient_acc, float *scratchpad);

void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, const NN_train_opt *opt) {
    if (opt->mini_batch_size < 1 || opt->mini_batch_size > train_len) {
        fprintf(stderr, "[ERROR]: mini_batch_size has to be in interval [1..train_len].");
        exit(1);
    }

    float error = FLT_MAX;
    size_t epoch = 0;

    /* Array for storing the intermediate products in the feed forward */
    size_t intermediate_products_len = nn->intermediate_activations_len;
    float *intermediate_products = nn_malloc(intermediate_products_len*sizeof(float));

    /* Array for storing deltas*/
    size_t deltas_len = 0;
    for (size_t i = 1; i < nn->units_configuration_len; ++i) {
        deltas_len += nn->units_configuration[i];
    }
    float *deltas = nn_malloc(deltas_len * sizeof(float));

    /* Array for accumulate gradients */
    float *gradient_acc = nn_malloc(nn->weights_len * sizeof(float));

    /*
    Scratchpad array for backprop.
    Length is the number of weights of the largest layer.
    */
    size_t largest_layer_size = 0;
    for (size_t i = 0; i < nn->layers_len - 1; ++i) {
        const size_t current_layer_len = (nn->units_configuration[i] + 1) * nn->units_configuration[i+1];
        if (current_layer_len > largest_layer_size) {
            largest_layer_size = current_layer_len;
        }
    }
    float *scratchpad = nn_malloc(largest_layer_size * sizeof(float));

    /* Array of index, that will be shuffled in order to do Stochastic and Mini-batch GD */
    size_t *train_indexes = nn_malloc(train_len * sizeof(size_t));
    for (size_t i = 0; i < train_len; ++i) {
        train_indexes[i] = i;
    }

    /* Getting output from intermediate activation */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    const size_t out_index = nn->intermediate_activations_len - out_len;
    const float *out = nn->intermediate_activations + out_index;

    while (error > opt->err_threshold) {

        /* Getting the current error, using the MSE */
        error = 0.0;
        for (size_t i = 0; i < train_len; ++i) {
            nn_feed_forward(nn, x_train + i*nn->units_configuration[0], NULL);

            for (size_t j = 0; j < out_len; ++j) {
                error += powf(y_train[i*out_len + j] - out[j], 2);
            }
        }
        error *= 1.0/(2.0*out_len*train_len);

        if (opt->err_epoch_logging >= 0) {
            if (epoch % opt->err_epoch_logging == 0) {
                printf("Error = %f\n", error);
            }
        }

        /*
        ==================================
        / Gradient Descent with backprop /
        ==================================
        */

        /* Shuffle train_indexes using Fisher–Yates shuffle algorithm (only if isn't used Batch GD) */
        if (opt->mini_batch_size != train_len) {
            for (size_t i = train_len - 1; i > 0; --i) {
                size_t j = rand() % (i+1);
                size_t tmp = train_indexes[i];
                train_indexes[i] = train_indexes[j];
                train_indexes[j] = tmp;
            }
        }

        for (size_t i = 0; i < train_len; i+=opt->mini_batch_size) {
            /* Reset gradient accumulator */
            memset(gradient_acc, 0, nn->weights_len * sizeof(float));

            /*
            (In the case of Mini-batch GD).
            If train_len is not divisible by opt->mini_batch_size, the last batch length is < opt->mini_batch_size.
            */
            const size_t current_batch_size = (i + opt->mini_batch_size - 1 >= train_len) ? (train_len - i) : opt->mini_batch_size;

            for (size_t batch_i = 0; batch_i < current_batch_size; ++batch_i) {
                size_t train_i = train_indexes[i + batch_i];

                nn_feed_forward(nn, x_train + train_i * nn->units_configuration[0], intermediate_products);
                backpropagation(nn, train_i, y_train, intermediate_products, intermediate_products_len, deltas, deltas_len, gradient_acc, scratchpad);
            }

            /* Update weights using the gradients */
            for (size_t j = 0; j < nn->weights_len; ++j) {
                nn->weights[j] -= opt->learning_rate * gradient_acc[j] * (1.0 / opt->mini_batch_size);
            }
        }
        epoch++;
    }

    free(intermediate_products);
    free(deltas);
    free(gradient_acc);
    free(scratchpad);
    free(train_indexes);
}

/*
Backpropagation is a gradient computation method (https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication).
*/
static void backpropagation(const NN *nn, size_t batch_i, const float *y_train, const float *intermediate_products, size_t intermediate_products_len, float *deltas, size_t deltas_len, float *gradient_acc, float *scratchpad) {

    /*
    We need to calculate all the delta(l) arrays (errors of the layer l) starting from the output layer.

    delta(L) (where L is the output layer) = '(a(L) - y_train(i)) .* f'(z(L))', where a(L) is the feed forward output.

    delta(l) (l from 1 to L-1) = 'transpose(nn->layers[l]) * delta(l+1) .* f'(z(l))',
    where z(l) is the intermediate product at layer l and f' the derivative of the activation of that layer.

    */

    /* delta(L) */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    size_t out_index = nn->intermediate_activations_len - out_len;
    float *out = nn->intermediate_activations + out_index;

    const size_t deltas_out_index = deltas_len - out_len;
    size_t intermediate_products_index = intermediate_products_len - out_len;
    for (size_t i = 0; i < out_len; ++i) {
        deltas[deltas_out_index + i] = out[i] - y_train[batch_i*out_len + i];
        deltas[deltas_out_index + i] *= nn->activations_derivative[nn->layers_len - 1](intermediate_products[intermediate_products_index+i]);
    }

    /* delta(L-1) .. delta(1) */
    size_t deltas_index = deltas_out_index;
    for (size_t i = nn->layers_len - 1; i > 0; --i) {
        const size_t res_len = nn->units_configuration[i] + 1;
        intermediate_products_index -= res_len;

        /* delta(l) = transpose(nn->layers[l]) * delta(l+1) .* f'(z(l) */
        nn_matrix_mul_t(
            deltas + deltas_index, 1, nn->units_configuration[i+1],
            nn->layers[i], nn->units_configuration[i] + 1, nn->units_configuration[i+1],
            scratchpad
        );

        for (size_t j = 1; j < res_len; ++j) {
            scratchpad[j] *= nn->activations_derivative[i-1](intermediate_products[intermediate_products_index+j]);
        }

        deltas_index -= res_len - 1;
        memcpy(deltas + deltas_index, scratchpad + 1, (res_len - 1)*sizeof(float));
    }

    /*
    Now we have all the delta(i) and we can calculate the gradient(l) (gradient of the layer l) for each l.

    gradient(l) = 'delta(l+1) * transpose(a(l))',
    where a(l) is the output ('f(z(l))') of the layer l.
    */

    size_t gradient_acc_index = 0;
    deltas_index = 0;
    float *activations_i = nn->intermediate_activations;
    for (size_t i = 0; i < nn->layers_len; ++i) {
        nn_matrix_mul(
            activations_i, nn->units_configuration[i] + 1, 1,
            deltas + deltas_index, 1, nn->units_configuration[i+1],
            scratchpad
        );

        /* Gradient accumulation */
        for (size_t j = 0; j < (nn->units_configuration[i] + 1) * nn->units_configuration[i+1]; ++j) {
            gradient_acc[gradient_acc_index + j] += scratchpad[j];
        }
        gradient_acc_index += (nn->units_configuration[i] + 1) * nn->units_configuration[i+1];

        activations_i += nn->units_configuration[i] + 1;
        deltas_index += nn->units_configuration[i+1];
    }
}
