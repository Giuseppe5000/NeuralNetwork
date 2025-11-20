#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stdio.h>

typedef struct NN NN; /* Opaque type */

enum Activation {
    NN_SIGMOID,
    NN_RELU,
    NN_TANH,
    NN_SOFTMAX, /* Usable only in the output layer! */
};

enum Weight_initialization {
    NN_UNIFORM,
    NN_GLOROT,
    NN_HE,
};

enum Loss_function {
    NN_CROSS_ENTROPY,
    NN_MSE,
};

typedef struct {
    float learning_rate;
    size_t epoch_num; /* Number of training epochs */

    /*
     *  FILE pointer where the loss through the epochs will be logged.
     *  If == 'NULL' no log occurs.
     *
     * The output files then can be plotted on gnuplot (and similar)
     * (see the examples).
     */
     FILE* loss_log_train_fp;
     FILE* loss_log_test_fp;

    /*
     *  How many training samples use at once for gradients update.
     *
     *  Values in [1..train_len].
     *  When '1'                     => Stochastic Gradient Descent.
     *  When '> 1' and '< train_len' => Mini-batch Gradient Descent.
     *  When '== train_len'          => Batch Gradient Descent.
     */
     size_t batch_size;

     enum Loss_function loss;
} NN_train_opt;

/*
*  Initialize the neural network, allocating the weights according to the 'units_configuration' and setting their value according to the stategy 'w_init'.
*  'units_activation' is an array of length 'units_configuration_len - 1' that specify the activation function for each layer.
*
*  [NOTE]: Bias terms are implicit.
*  [NOTE]: Obviously the freeing of NN* is responsibility the user (using 'nn_free').
*
*  Example:
*      size_t units_configuration[] = {3, 2, 1};
*      enum Activation units_activation[] = {NN_TANH, NN_SIGMOID};
*
*                  b0
*                       b1
*               |> x0
*               |     > x3
*      input => |> x1      > x5 => output
*               |     > x4   |
*               |> x2   |    |
*                       |    |
*                     TANH  SIGMOID
*
*      Here we have a (fully connected) 2-layer neural network (or a 1-hidden-layer nn).
*
*      x0, x1, x2 are the input and x3, x4, x5 are neurons.
*      b0 and b1 are always 1, their weights are the biases (encoded with all the other weights).
*
*      The first matrix of weigths is of size 4x2 (3x2 input weights and a 1x2 for bias)
*      and the second is 3x1 (2x1 for input weights and 1x1 for bias).
*
*      first_matrix_weights = {
*          w(b0<->x3), w(b0<->x4),   <-- bias
*          w(x0<->x3), w(x0<->x4),
*          w(x1<->x3), w(x1<->x4),
*          w(x2<->x3), w(x2<->x4),
*      }
*
*      second_matrix_weights = {
*          w(b1<->x5),    <-- bias
*          w(x3<->x5),
*          w(x4<->x5),
*      }
*/
NN *nn_init(const size_t *units_configuration, size_t units_configuration_len, const enum Activation *units_activation, enum Weight_initialization w_init);

/*
*  Freeing allocated memory.
*
*  [NOTE]: Using 'nn' after calling this function, obviously, causes undefined behaviour.
*/
void nn_free(NN *nn);

/*
*  Train the neural network using train data (features are in x_* and correct labels in y_*)
*  and checks also test data loss (if 'loss_log_test_fp' != NULL).
*
*  'x_train' has to be a matrix of 'train_len' rows and 'units_configuration[0]' columns.
*  'y_train' has to be a matrix of 'train_len' rows and 'units_configuration[units_configuration_len - 1]' columns.
*
*  'x_test' has to be a matrix of 'test_len' rows and 'units_configuration[0]' columns.
*  'y_test' has to be a matrix of 'test_len' rows and 'units_configuration[units_configuration_len - 1]' columns.
*/
void nn_fit(NN *nn, const float *x_train, const float *y_train, size_t train_len, const float *x_test, const float* y_test, size_t test_len, const NN_train_opt *opt);

/*
*  Feed forward the network through the layers.
*
*  'x' has to be an array of length 'units_configuration[0]'.
*  The prediction result will be put in the 'out' array of length 'units_configuration[units_configuration_len - 1]'.
*/
void nn_predict(NN *nn, const float *x, float *out);

#endif /* NEURAL_NETWORK */
