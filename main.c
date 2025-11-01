#include <stdio.h>
#include "neural_network.h"

int main(void) {
    size_t units_configuration[] = {3, 2, 1};
    size_t units_configuration_len = sizeof(units_configuration) / sizeof(units_configuration[0]);

    enum Activation units_activation[] = {NN_SIGMOID, NN_TANH, NN_RELU};

    NN * nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    /* Train */

    const float x_train[] = {
        1, 2, 3,
        1, 1, 3,
        1, 2, 5,
        1, 0, 3,
    };

    const float y_train[] = {
        1,
        0,
        0,
        0,
    };

    size_t train_len = sizeof(x_train) / sizeof(x_train[0]) / units_configuration[0];
    nn_fit(nn, x_train, y_train, train_len, 0.1, 0.001);

    /* Test */
    const float x[] = {1, 2, 3};
    float out[1];

    nn_predict(nn, x, out);

    printf("OUT = [%f]\n", *out);

    nn_free(nn);
    return 0;
}
