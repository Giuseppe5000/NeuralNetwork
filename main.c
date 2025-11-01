#include <stdio.h>
#include "neural_network.h"

int main(void) {
    size_t units_configuration[] = {3, 2, 1};
    size_t units_configuration_len = sizeof(units_configuration) / sizeof(units_configuration[0]);

    enum Activation units_activation[] = {NN_SIGMOID, NN_TANH, NN_RELU};

    NN * nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    nn_free(nn);
    return 0;
}
