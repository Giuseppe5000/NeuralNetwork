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
};

/*
Initialize the nn, allocating the weights according to the 'units_configuration' and setting their value according to the stategy 'w_init'.
'units_activation' is an array of length 'units_configuration_len' that specify the activation function for each layer.

Example:
    size_t units_configuration[] = {3, 2, 1};
    enum Activation units_activation[] = {NN_SIGMOID, NN_TANH, NN_RELU};

             |> x0
             |     > x3
    input => |> x1      > x5 => output
             |     > x4
             |> x2

    (The xn's are neurons, and x0/x1/x2 are the input).
    Here we have a (fully connected) 2-layer neural network (or a 1-hidden-layer nn).

    The first matrix of weigths is of size 3x2 and the second is 2x1.

    first_matrix_weights = {
        w(x0<->x3), w(x0<->x4),
        w(x1<->x3), w(x1<->x4),
        w(x2<->x3), w(x2<->x4),
    }

    second_matrix_weights = {
        w(x3<->x5),
        w(x4<->x5),
    }
*/
NN *NN_init(size_t *units_configuration, size_t units_configuration_len, enum Activation *units_activation, enum Weight_initialization w_init);

/*
Freeing allocated memory.

[NOTE]: Using 'nn' after calling this function, obviously, causes undefined behaviour.
*/
void NN_free(NN *nn);

/*
Train the neural network.

'x_train' is a matrix of 'train_len' rows and 'units_configuration[0]' columns.
'y_train' is a matrix of 'train_len' rows and 'units_configuration[units_configuration_len - 1]' columns.
*/
void NN_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, float learning_rate, float err_threshold);

/*
Feed forward the network.

'x' is an array of length 'units_configuration[0]'.
The prediction result will be put in the 'res' array of length 'units_configuration[units_configuration_len - 1]'.
*/
void NN_predict(NN *nn, const float *x, float *res);

#endif /* NEURAL_NETWORK */
