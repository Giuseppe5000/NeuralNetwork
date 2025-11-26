#include "../include/neural_network.h"
#include "nn_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ======================== Data structures ======================== */

typedef void (*nn_activation)(float *, float *, size_t);
typedef float (*nn_activation_derivative)(float);

struct NN {
    /*
    *  NN architecture configuration.
    */
    size_t *units_configuration;
    size_t units_configuration_len;

    /*
    *  All the weights of the network, stored as a linear array of floats.
    *
    *  'weights_len' is equal to the sum of each subsequent pair of 'units_configuration' (adding the biases obv).
    *  Example:
    *      size_t units_configuration[] = {3, 2, 1};
    *
    *      Then 'weights_len' = (3+1) * 2 + (2+1) * 1  = 8 + 3 = 11.
    */
    float *weights;
    size_t weights_len;

    /*
    *  'layers' is a view of the 'weights'.
    *  It is an array of pointer, each pointer points to the start of a matrix in 'weights'.
    *
    *  For example:
    *    layers[0] = pointer to the first element of first layer.
    *    layers[1] = pointer to the first element of second layer.
    *
    *  This is convinient for moving between the nn layers.
    *
    *  'layers_len' = 'units_configuration_len' - 1.
    */
    float **layers;
    size_t layers_len;

    /*
    *  Array of activation functions (and its derivatives), one for each layers.
    *  length = 'layers_len' = 'units_configuration_len' - 1.
    */
    nn_activation *activations;
    nn_activation_derivative *activations_derivative;

    /*
    *  Array for storing the intermediate activations of the feed forward (contains even input and output).
    *  Storing this activation is useful for the training (in backpropagation).
    *
    *  Its length has to be the sum of the elements of nn->units_configuration + the biases:
    *  so, (nn->units_configuration[0] + 1) +
    *      (nn->units_configuration[1] + 1) +
    *                  .... +
    *      (nn->units_configuration[nn->units_configuration_len - 1]).
    */
    float *intermediate_activations;
    size_t intermediate_activations_len;
};

/* ================================================================= */


/* ===================== Forward declarations ====================== */

static void nn_feed_forward(NN *nn, const float *x);

static void backpropagation(
    const NN *nn,
    size_t batch_i,
    const float *y_train,
    float *deltas,
    size_t deltas_len,
    float *gradient_acc,
    float *scratchpad,
    enum Loss_function loss
);

/* ================================================================= */


/* ======================= Helper functions ======================== */

/*
*  Wrapping malloc with this helper function,
*  handling OOM with error logging and exit.
*/
static void *nn_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "[ERROR]: Out of memory.\n");
        exit(1);
    }
    return ptr;
}

/*
*  Returns a random float between 'min' and 'max'.
*/
static float randf(float min, float max) {
    return min + (max - min) * ((float)rand()/(float)RAND_MAX);
}

/*
*  Glorot weight initialization.
*  (https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization).
*/
static float glorot(size_t fan_in, size_t fan_out) {
    const float x = sqrtf(6.0 / (fan_out + fan_in));
    return randf(-x, x);
}

/*
*  He weight initialization, using Box-Muller transform.
*  (https://en.wikipedia.org/wiki/Weight_initialization#He_initialization).
*/
static float he(size_t fan_in) {
    const float stddev = sqrtf(2.0 / fan_in);
    const float u1 = (float)rand() / (float)RAND_MAX;
    const float u2 = (float)rand() / (float)RAND_MAX;
    const float z0 = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * M_PI * u2);
    return z0 * stddev;
}

/*
*  Multiply the matrixes 'A' and 'B', putting the result in 'res' (naive implementation).
*  'res' has to be a matrix of 'A_rows' rows and 'B_cols' columns.
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
*  Multiply the matrixes 'A' and the transpose of 'B' directly, putting the result in 'res'.
*  'res' needs to be a matrix of 'A_rows' rows and 'B_rows' columns.
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

/*
*  Computes the loss using the selected loss function ('loss')
*  and using the data in 'x_data' and 'y_data' of length 'data_len'.
*  The computed loss will be printed into 'fp'.
*/
static void loss_log(NN *nn, FILE *fp, enum Loss_function loss_type, const float *x_data, const float *y_data, size_t data_len) {
    /* Getting output from intermediate activation */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    const size_t out_index = nn->intermediate_activations_len - out_len;
    const float *out = nn->intermediate_activations + out_index;

    float loss = 0.0;

    switch(loss_type) {
        case NN_CROSS_ENTROPY:
            for (size_t i = 0; i < data_len; ++i) {
                nn_feed_forward(nn, x_data + i*nn->units_configuration[0]);
                float loss_i = 0.0;

                /* If the nn has a single output we need to use this slightly different formula */
                if (out_len == 1) {
                    loss_i = y_data[i*out_len] * logf(out[0]) + (1 - y_data[i*out_len]) * logf(1 - out[0]);
                } else {
                    for (size_t j = 0; j < out_len; ++j) {
                        loss_i += y_data[i*out_len + j] * logf(out[j]);
                    }
                }

                loss_i *= -1;
                loss += loss_i;
            }
            loss *= 1.0/(data_len);
            break;

        case NN_MSE:
            for (size_t i = 0; i < data_len; ++i) {
                nn_feed_forward(nn, x_data + i*nn->units_configuration[0]);

                for (size_t j = 0; j < out_len; ++j) {
                    loss += powf(y_data[i*out_len + j] - out[j], 2);
                }
            }
            loss *= 1.0/(2.0*out_len*data_len);
            break;

        default:
            fprintf(stderr, "[ERROR]: Invalid loss function.\n");
            exit(1);
    }

    fprintf(fp, "%f ", loss);
}

/* ================================================================= */


/* ============== Activation functions and derivative ============== */

static void sigmoid_vec(float *x, float *out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i] = 1.0 / (1.0 + expf(-x[i]));
    }
}

static void relu_vec(float *x, float *out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i] = (x[i] > 0.0) ? x[i] : 0.0;
    }
}

static void tanh_vec(float *x, float *out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        out[i] = tanhf(x[i]);
    }
}

/*
*  Safe implementation.
*  (https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms)
*  (https://en.wikipedia.org/wiki/Softmax_function#Example)
*/
static void softmax(float *x, float *out, size_t len) {

    /* Get the max of 'x' */
    float max = -FLT_MAX;
    for (size_t i = 0; i < len; ++i) {
        if (x[i] > max) max = x[i];
    }

    /*
    *  Get the sum of all 'exp(beta*(xi - max))'.
    *  beta = 1.0.
    */
    float sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum += expf((x[i] - max));
    }

    /* Compute the softmax for each xi*/
    for (size_t i = 0; i < len; ++i) {
        out[i] = expf((x[i] - max)) / sum;
    }
}

/* Computes the derivative taking as input the sigmoid of x */
static float sigmoid_derivative(float sigmoid_x) {
    return sigmoid_x * (1.0 - sigmoid_x);
}

/* Computes the derivative taking as input the relu of x */
static float relu_derivative(float x_relu) {
    /* Not derivable when x = 0, so for practical reasons the value in x = 0 is 0 */
    return (x_relu > 0.0) ? 1.0 : 0.0;
}

/* Computes the derivative taking as input the tanh of x */
static float tanh_derivative(float x_tanh) {
    return 1.0 - powf(x_tanh, 2);
}

/* ================================================================= */



NN *nn_init(const size_t *units_configuration, size_t units_configuration_len, const enum Activation *units_activation, enum Weight_initialization w_init) {
    NN *nn = nn_malloc(sizeof(NN));

    /* CUDA init */
    NN_CUDA_ctx ctx = {0};
    nn_cuda_init(&ctx);

    /* Copy units_configuration into the struct */
    nn->units_configuration = nn_malloc(units_configuration_len * sizeof(size_t));
    nn->units_configuration_len = units_configuration_len;
    memcpy(nn->units_configuration, units_configuration, units_configuration_len * sizeof(size_t));

    /* Weights and layers initialize */
    nn->weights_len = 0;

    for (size_t i = 0; i < units_configuration_len - 1; ++i) {
        /*
        *  (units_configuration[i] * units_configuration[i+1]) = N x M matrix weights
        *  units_configuration[i+1] = weights for bias term
        */
        nn->weights_len += (units_configuration[i] * units_configuration[i+1]) + units_configuration[i+1];
    }
    nn_cuda_malloc(nn->weights_len * sizeof(float), &nn->weights);

    nn->layers_len = units_configuration_len - 1;
    nn->layers = nn_malloc(nn->layers_len * sizeof(float *));
    size_t counter = 0;
    for (size_t i = 0; i < nn->layers_len; ++i) {
        nn->layers[i] = nn->weights + counter;
        counter += (units_configuration[i] * units_configuration[i+1]) + units_configuration[i+1];
    }

    srand(time(NULL));
    for(size_t i = 0; i < nn->layers_len; ++i) {
        const size_t layer_size = (units_configuration[i] * units_configuration[i+1]) + units_configuration[i+1];

        for (size_t j = 0; j < layer_size; ++j) {
            float *weight = nn->layers[i] + j;

            switch (w_init) {
                case NN_UNIFORM:
                    *weight = randf(-0.5,0.5);
                    break;
                case NN_GLOROT:
                    *weight = glorot(units_configuration[i] + 1, units_configuration[i+1]);
                    break;
                case NN_HE:
                    *weight = he(units_configuration[i] + 1);
                    break;
                default:
                    fprintf(stderr, "[ERROR]: Invalid weights init method.\n");
                    exit(1);
            }
        }
    }

    /* Activation functions initialize */
    nn->activations = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation));
    nn->activations_derivative = nn_malloc((units_configuration_len - 1) * sizeof(nn_activation_derivative));

    for (size_t i = 0; i < units_configuration_len - 1; ++i) {
        switch (units_activation[i]) {
            case NN_SIGMOID:
                (nn->activations)[i] = sigmoid_vec;
                (nn->activations_derivative)[i] = sigmoid_derivative;
                break;
            case NN_RELU:
                (nn->activations)[i] = relu_vec;
                (nn->activations_derivative)[i] = relu_derivative;
                break;
            case NN_TANH:
                (nn->activations)[i] = tanh_vec;
                (nn->activations_derivative)[i] = tanh_derivative;
                break;
            case NN_SOFTMAX:
                if (i != nn->layers_len - 1) {
                    fprintf(stderr, "[ERROR]: NN_SOFTMAX can be used only on the output layer.\n");
                    exit(1);
                }
                (nn->activations)[i] = softmax;
                (nn->activations_derivative)[i] = NULL; /* Not needed */
                break;
            default:
                fprintf(stderr, "[ERROR]: Check if the 'units_activation' length == units_configuration_len - 1.\n");
                exit(1);
        }
    }

    /* Intermediate activations initialize */
    nn->intermediate_activations_len = 0;

    for (size_t i = 0; i < nn->units_configuration_len; ++i) {
        nn->intermediate_activations_len += nn->units_configuration[i] + 1; /* Number of neurons + bias */
    }
    nn->intermediate_activations_len--; /* Last layer doesn't have bias */

    nn_cuda_malloc(nn->intermediate_activations_len * sizeof(float), &nn->intermediate_activations);

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

void nn_predict(NN *nn, const float *x, float *out) {
    nn_feed_forward(nn, x);

    /* Copy the output, which is stored in the last elements of intermediate activations into 'out' */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    const size_t out_index = nn->intermediate_activations_len - out_len;
    memcpy(out, nn->intermediate_activations + out_index, out_len*sizeof(float));
}

/*
*  Feed forward the NN.
*
*  'x' is an array of length 'nn->units_configuration[0]'.
*
*  Each output of the feed forward is computed and stored in 'nn->intermediate_activations' (input included).
*  So the final output of the network is stored at the end of this array.
*/
static void nn_feed_forward(NN *nn, const float *x) {
    const size_t x_rows = 1;
    size_t x_cols = nn->units_configuration[0];

    /* Copy input into intermediate_activations */
    nn->intermediate_activations[0] = 1.0; /* Bias */
    memcpy(nn->intermediate_activations + 1, x, x_cols * sizeof(float));

    /* Feed forward through the nn layers */
    float *activations_i = nn->intermediate_activations; /* Activations of units i */
    float *activations_i_next = nn->intermediate_activations + x_cols + 1; /* Activations of units i+1 */

    for (size_t i = 0; i < nn->layers_len; ++i) {
        const size_t res_len = nn->units_configuration[i+1];

        /*
        *  We need to know we we are on the last layer.
        *  Because in that case we don't need to put 1.0 in activations_i_next
        *  for biases multiplication.
        */
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

        /* Applying activation function */
        nn->activations[i](
            activations_i_next + is_not_last_layer,
            activations_i_next + is_not_last_layer,
            res_len
        );

        /* Now the "input" is the 'activation_i = activation_i_next' and we move forward 'activation_i_next' */
        activations_i = activations_i_next;
        activations_i_next += res_len + is_not_last_layer;
        x_cols = nn->units_configuration[i+1];
    }
}

void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, const float *x_test, const float* y_test, size_t test_len, const NN_train_opt *opt) {
    if (opt->batch_size < 1 || opt->batch_size > train_len) {
        fprintf(stderr, "[ERROR]: mini_batch_size has to be in interval [1..train_len].\n");
        exit(1);
    }

    if (opt->loss_type == NN_MSE && nn->activations[nn->layers_len - 1] == softmax) {
        fprintf(stderr, "[ERROR]: You should use NN_CROSS_ENTROPY with softmax activation.\n");
        exit(1);
    }


    /*
    *  ==========================
    *  || Initial memory setup ||
    *  ==========================
    */

    /* Array for storing deltas (for backprop) */
    size_t deltas_len = 0;
    for (size_t i = 1; i < nn->units_configuration_len; ++i) {
        deltas_len += nn->units_configuration[i];
    }
    float *deltas = nn_malloc(deltas_len * sizeof(float));

    /* Array for accumulating gradients */
    float *gradient_acc = nn_malloc(nn->weights_len * sizeof(float));

    /*
    *  Scratchpad array big enough used in backprop.
    *  Big enough = the layer with the biggest amout of weights.
    *  It is useful for storing temporary outputs.
    */
    size_t largest_layer_size = 0;
    for (size_t i = 0; i < nn->layers_len; ++i) {
        const size_t current_layer_len = (nn->units_configuration[i] + 1) * (nn->units_configuration[i+1]);
        if (current_layer_len > largest_layer_size) {
            largest_layer_size = current_layer_len;
        }
    }
    float *scratchpad = nn_malloc(largest_layer_size * sizeof(float));

    /* Array of train index, that will be shuffled in order to do Stochastic and Mini-batch GD */
    size_t *train_indexes = nn_malloc(train_len * sizeof(size_t));
    for (size_t i = 0; i < train_len; ++i) {
        train_indexes[i] = i;
    }

    /* If logging is enabled, print some informations on the top of the file */
    if (opt->loss_log_fp != NULL) {
        fprintf(opt->loss_log_fp, "# COLS INFO: (epoch) (train_loss) (test_loss)\n");
        fflush(opt->loss_log_fp);
    }


    /*
    *  =================
    *  || Epochs loop ||
    *  =================
    */

    for (size_t epoch = 0; epoch < opt->epochs; ++epoch) {

        /*
        *  (If loss log is enabled)
        *  Compute the error (using the selected loss function)
        *  and write it in the log.
        */
        if (opt->loss_log_fp != NULL) {
            fprintf(opt->loss_log_fp, "%zu ", epoch);
            loss_log(nn, opt->loss_log_fp, opt->loss_type, x_train, y_train, train_len);
            loss_log(nn, opt->loss_log_fp, opt->loss_type, x_test, y_test, test_len);
            fprintf(opt->loss_log_fp, "\n");
            fflush(opt->loss_log_fp); /* TODO: not a good idea */
        }

        /* Shuffle train_indexes using Fisher–Yates shuffle algorithm (only if isn't used Batch GD) */
        if (opt->batch_size != train_len) {
            for (size_t i = train_len - 1; i > 0; --i) {
                const size_t j = rand() % (i+1);
                const size_t tmp = train_indexes[i];
                train_indexes[i] = train_indexes[j];
                train_indexes[j] = tmp;
            }
        }

        /* Effective training */
        for (size_t i = 0; i < train_len; i+=opt->batch_size) {

            /* Reset gradient accumulator */
            memset(gradient_acc, 0, nn->weights_len * sizeof(float));

            /*
            *  (Edge case in Mini-batch GD).
            *  If train_len is not divisible by opt->mini_batch_size, the last batch length is < opt->mini_batch_size.
            */
            const size_t current_batch_size = (i + opt->batch_size - 1 >= train_len) ? (train_len - i) : opt->batch_size;

            for (size_t batch_i = 0; batch_i < current_batch_size; ++batch_i) {
                const size_t train_i = train_indexes[i + batch_i];

                nn_feed_forward(nn, x_train + train_i * nn->units_configuration[0]);
                backpropagation(nn, train_i, y_train, deltas, deltas_len, gradient_acc, scratchpad, opt->loss_type);
            }

            /* Update weights using the gradients */
            for (size_t j = 0; j < nn->weights_len; ++j) {
                nn->weights[j] -= opt->learning_rate * gradient_acc[j] * (1.0 / current_batch_size);
            }
        }
    }

    free(deltas);
    free(gradient_acc);
    free(scratchpad);
    free(train_indexes);
}

/*
*  Backpropagation is a gradient computation method (https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication).
*/
static void backpropagation(const NN *nn, size_t batch_i, const float *y_train, float *deltas, size_t deltas_len, float *gradient_acc, float *scratchpad, enum Loss_function loss) {

    /*
    *  We need to calculate all the delta(l) arrays (errors of the neurons at layer l) starting from the output.
    *  We start from the computation of the last delta and we use it for computing the previous, and so on.
    *
    *
    *  delta(L) (where L is the output layer) = '' (a(L) - y_train) .* f'(z(L)) '',
    *  where:
    *    - a(L) is the output (activation) of the layer L.
    *    - z(L) is the preactivation of a(L).
    *    - f' is the derivative of the activation function in that layer.
    *
    *
    *  After getting delta(L) we can compute all the other deltas using this formula:
    *  delta(l) (l from 1 to L-1) = '' transpose(nn->layers[l]) * delta(l+1) .* f'(z(l)) ''.
    *  [NOTE]: delta(0) (the input) is not included.
    *
    *
    *  By the way, we can compute f'(z(l)) using the directly activation a(l), so there is no need of storing all z(l).
    *  In fact we defined the activations derivatives in terms of the function itself,
    *  so we can just do f'(a(l)), where f' is not the real derivative of the function but the derivative in terms of the "primitive".
    */

    /* delta(L) */
    const size_t out_len = nn->units_configuration[nn->units_configuration_len - 1];
    const size_t out_index = nn->intermediate_activations_len - out_len;
    float *out = nn->intermediate_activations + out_index;

    const size_t deltas_out_index = deltas_len - out_len;
    size_t intermediate_activations_index = nn->intermediate_activations_len - out_len;
    for (size_t i = 0; i < out_len; ++i) {
        deltas[deltas_out_index + i] = out[i] - y_train[batch_i*out_len + i];
        if (loss == NN_MSE) {
            deltas[deltas_out_index + i] *= nn->activations_derivative[nn->layers_len - 1](nn->intermediate_activations[intermediate_activations_index+i]);
        }
    }

    /* delta(L-1) .. delta(1) */
    size_t deltas_index = deltas_out_index;
    for (size_t i = nn->layers_len - 1; i > 0; --i) {
        const size_t res_len = nn->units_configuration[i] + 1;
        intermediate_activations_index -= res_len;

        /* delta(l) = transpose(nn->layers[l]) * delta(l+1) .* f'(z(l) */
        nn_matrix_mul_t(
            deltas + deltas_index, 1, nn->units_configuration[i+1],
            nn->layers[i], nn->units_configuration[i] + 1, nn->units_configuration[i+1],
            scratchpad
        );

        for (size_t j = 1; j < res_len; ++j) {
            scratchpad[j] *= nn->activations_derivative[i-1](nn->intermediate_activations[intermediate_activations_index+j]);
        }

        deltas_index -= res_len - 1;
        memcpy(deltas + deltas_index, scratchpad + 1, (res_len - 1) * sizeof(float));
    }

    /*
    *  Now we have all the delta(i) and we can calculate the gradient(l) (gradient of the layer l) for each l.
    *
    *  The formula is:
    *  gradient(l) = 'a(l) * delta(l+1)',
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
