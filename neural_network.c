#include "neural_network.h"

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
