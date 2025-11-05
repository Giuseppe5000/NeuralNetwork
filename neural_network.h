#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stddef.h>

typedef struct NN NN; /* Opaque type */

enum Activation {
    NN_SIGMOID,
    NN_RELU,
    NN_TANH,
};

enum Weight_initialization {
    NN_UNIFORM,
    NN_GLOROT,
    NN_HE,
};

typedef struct {
    float learning_rate;
    float err_threshold;

    /*
    After how much epochs the error will be logged.
    For negative numbers no logging will occur.
    */
    int err_epoch_logging;

    size_t mini_batch_size;
} NN_train_opt;

/*
Initialize the nn, allocating the weights according to the 'units_configuration' and setting their value according to the stategy 'w_init'.
'units_activation' is an array of length 'units_configuration_len' that specify the activation function for each layer.

[NOTE]: Bias terms are implicit.

Example:
    size_t units_configuration[] = {3, 2, 1};
    enum Activation units_activation[] = {NN_SIGMOID, NN_TANH, NN_RELU};

                b0
             |> x0   b1
             |     > x3
    input => |> x1      > x5 => output
             |     > x4
             |> x2

    (The xn's are neurons, and x0/x1/x2 are the input).
    (b0 and b1 are always 1).
    Here we have a (fully connected) 2-layer neural network (or a 1-hidden-layer nn).

    The first matrix of weigths is of size 4x2 (3x2 input weights and a 1x2 for bias)
    and the second is 3x1 (2x1 for input weights and 1x1 for bias).

    first_matrix_weights = {
        w(b0<->x3), w(b0<->x4),   <-- bias
        w(x0<->x3), w(x0<->x4),
        w(x1<->x3), w(x1<->x4),
        w(x2<->x3), w(x2<->x4),
    }

    second_matrix_weights = {
        w(b1<->x5),    <-- bias
        w(x3<->x5),
        w(x4<->x5),
    }
*/
NN *nn_init(size_t *units_configuration, size_t units_configuration_len, enum Activation *units_activation, enum Weight_initialization w_init);

/*
Freeing allocated memory.

[NOTE]: Using 'nn' after calling this function, obviously, causes undefined behaviour.
*/
void nn_free(NN *nn);

/*
Train the neural network.

'x_train' is a matrix of 'train_len' rows and 'units_configuration[0]' columns.
'y_train' is a matrix of 'train_len' rows and 'units_configuration[units_configuration_len - 1]' columns.
*/
void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, const NN_train_opt *opt);

/*
Feed forward the network.

'x' is an array of length 'units_configuration[0]'.
The prediction result will be put in the 'res' array of length 'units_configuration[units_configuration_len - 1]'.
*/
void nn_predict(NN *nn, const float *x, float *out);

#endif /* NEURAL_NETWORK */
