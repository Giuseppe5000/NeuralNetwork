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
    const float u1 = ((float) rand()) / (RAND_MAX);
    const float u2 = ((float)rand()) / (RAND_MAX);
    const float z0 = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * M_PI * u2);
    return z0 * stddev;
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
            res[row*B_cols + col] = 0;
            for (size_t i = 0; i < A_cols; ++i) {
                res[row*B_cols + col] += A[row*A_cols + i] * B[i*B_cols + col];
            }
        }
    }
}

/*
Puts into 'res' (matrix of 'A_col' rows and 'A_rows' cols) the transpose of 'A'.
*/
static void nn_matrix_transpose(const float *A, size_t A_rows, size_t A_cols, float *res) {
    const size_t res_cols = A_rows;

    for (size_t row = 0; row < A_rows; ++row) {
        for (size_t col = 0; col < A_cols; ++col) {
            res[col*res_cols + row] = A[row*A_cols + col];
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

/*
Feed forward the NN.

'x' is an array of length 'nn->units_configuration[0]'.
The prediction result will be put in the 'res' array of length 'units_configuration[nn->units_configuration_len - 1]'.

The intermediate products will be put in the 'intermediate_products' array IF it is not NULL.
Its length has to be the sum of the elements of nn->units_configuration + the biases:
    so, (nn->units_configuration[0] + 1) +  (nn->units_configuration[1] + 1) + .. + (nn->units_configuration[nn->units_configuration_len - 1]).
*/
static void nn_feed_forward(NN *nn, const float *x, float *out, float *intermediate_products) {
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

    float input[max_neurons + 1];
    input[0] = 1.0; /* Bias */
    memcpy(input + 1, x, x_cols * sizeof(float));

    /* Adding input as first intermidiate product */
    size_t intermediate_products_counter = 0;
    if (intermediate_products != NULL) {
        for (size_t i = 0; i < nn->units_configuration[0] + 1; ++i) {
            intermediate_products[intermediate_products_counter++] = input[i];
        }
    }

    /* Feed forward through the nn layers */
    for (size_t i = 0; i < nn->layers_len; ++i) {
        size_t res_len = nn->units_configuration[i+1];
        float res[res_len];

        nn_matrix_mul(
            input, x_rows, x_cols + 1, /* + 1 is for multiply the bias weights */
            nn->layers[i], nn->units_configuration[i] + 1, nn->units_configuration[i+1],
            res
        );

        /* Saving intermediate products */
        if (intermediate_products != NULL) {

            /* If we are on an itermediate layer then we have to add the bias terms */
            if (i != nn->layers_len - 1) {
                intermediate_products[intermediate_products_counter++] = 1.0;
            }

            for (size_t j = 0; j < res_len; ++j) {
                intermediate_products[intermediate_products_counter++] = res[j];
            }
        }

        /* Applying activation function */
        for (size_t j = 0; j < res_len; ++j) {
            res[j] = nn->activations[i](res[j]);
        }

        /* 'res' is the new input */
        memcpy(input + 1, res, res_len * sizeof(float));
        x_cols = nn->units_configuration[i+1];
    }

    memcpy(out, input + 1, nn->units_configuration[nn->units_configuration_len - 1] * sizeof(float));
}

void nn_predict(NN *nn, const float *x, float *out) {
    nn_feed_forward(nn, x, out, NULL);
}

void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, const NN_train_opt *opt) {
    float error = FLT_MAX;
    size_t epoch = 0;

    /* Array for storing the intermediate products in the feed forward */
    size_t intermediate_products_len = 0;
    for (size_t i = 0; i < nn->units_configuration_len; ++i) {
        intermediate_products_len += nn->units_configuration[i] + 1; /* Number of neurons + bias */
    }
    intermediate_products_len--; /* Last layer don't have bias */
    float intermediate_products[intermediate_products_len];

    /* Array for storing the output */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    float out[out_len];

    /* Array for storing deltas*/
    size_t deltas_len = 0;
    for (size_t i = 1; i < nn->units_configuration_len; ++i) {
        deltas_len += nn->units_configuration[i];
    }
    float deltas[deltas_len];

    /* Array for accumulate gradients */
    float *gradient_acc = nn_malloc(nn->weights_len * sizeof(float));


    while (error > opt->err_threshold) {

        /* Getting the current error, using the MSE */
        error = 0.0;
        for (size_t i = 0; i < train_len; ++i) {
            nn_predict(nn, x_train + i*nn->units_configuration[0], out);

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
        =================
        / Mini batch GD / (NOT YET)
        =================
        */

        /* Reset gradient accumulator */
        memset(gradient_acc, 0, nn->weights_len * sizeof(float));

        for (size_t batch_i = 0; batch_i < opt->mini_batch_size; ++batch_i) {

            /*
            ================
            / Feed forward /
            ================
            */

            nn_feed_forward(nn, x_train + batch_i * nn->units_configuration[0], out, intermediate_products);

            /*
            ===================
            / Backpropagation /
            ===================
            */

            /*
            Backpropagation is a gradient computation method (https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication).

            We need to calculate all the delta(l) arrays (errors of the layer l) starting from the output layer.

            delta(L) (where L is the output layer) = '(a(L) - y_train(i)) .* f'(z(L))', where a(L) is the feed forward output.

            delta(l) (l from 1 to L-1) = 'transpose(nn->layers[l]) * delta(l+1) .* f'(z(l))',
            where z(l) is the intermediate product at layer l and f' the derivative of the activation of that layer.
            */

            /* delta(L) */
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
                float res[res_len];
                intermediate_products_index -= res_len;

                /* nn->layers[i] + nn->units_configuration[i+1] is for skipping bias weights, delta do not have to be calculated for them */
                float layers_transpose[nn->units_configuration[i] * nn->units_configuration[i+1]];
                nn_matrix_transpose(
                    nn->layers[i],
                    nn->units_configuration[i] + 1,
                    nn->units_configuration[i+1],
                    layers_transpose
                );

                nn_matrix_mul(
                    deltas + deltas_index, 1, nn->units_configuration[i+1],
                    layers_transpose, nn->units_configuration[i+1], nn->units_configuration[i] + 1,
                    res
                );

                for (size_t j = 1; j < res_len; ++j) {
                    res[j] *= nn->activations_derivative[i-1](intermediate_products[intermediate_products_index+j]);
                }

                deltas_index -= res_len - 1;
                memcpy(deltas + deltas_index, res + 1, (res_len - 1)*sizeof(float));
            }

            /*
            Now we have all the delta(i) and we can calculate the gradient(l) (gradient of the layer l) for each l.

            gradient(l) = 'delta(l+1) * transpose(a(l))',
            where a(l) is the output ('f(z(l))') of the layer l.
            */

            size_t counter_index = 0;
            size_t gradient_acc_index = 0;
            deltas_index = 0;
            for (size_t i = 0; i < nn->layers_len; ++i) {
                float intermediate_activations_i[nn->units_configuration[i] + 1];

                for (size_t j = 0; j < nn->units_configuration[i] + 1; ++j) {
                    /*
                    if counter_index == 0 then the first activation a(l) is just the input,
                    and so it does not have an activation function
                    */
                    if (counter_index == 0) {
                        intermediate_activations_i[j] = intermediate_products[counter_index+j];
                    } else {
                        if (j == 0) { /* Bias term doesn't have activation function */
                            intermediate_activations_i[j] = intermediate_products[counter_index+j];
                        }
                        else {
                            intermediate_activations_i[j] = nn->activations[i-1](intermediate_products[counter_index+j]);
                        }
                    }
                }

                float res[(nn->units_configuration[i] + 1) * nn->units_configuration[i+1]];

                nn_matrix_mul(
                    intermediate_activations_i, nn->units_configuration[i] + 1, 1,
                    deltas + deltas_index, 1, nn->units_configuration[i+1],
                    res
                );

                /* Gradient accumulation */
                for (size_t j = 0; j < (nn->units_configuration[i] + 1) * nn->units_configuration[i+1]; ++j) {
                    gradient_acc[gradient_acc_index + j] += res[j];
                }
                gradient_acc_index += (nn->units_configuration[i] + 1) * nn->units_configuration[i+1];

                counter_index += nn->units_configuration[i] + 1;
                deltas_index += nn->units_configuration[i+1];
            }
        }

        /* Update weights */
        for (size_t j = 0; j < nn->weights_len; ++j) {
            nn->weights[j] -= opt->learning_rate * gradient_acc[j] * (1.0 / opt->mini_batch_size);
        }

        epoch++;
    }

    free(gradient_acc);
}
