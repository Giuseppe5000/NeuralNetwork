#include <stdio.h>
#include "../neural_network.h"

int main(void) {
    size_t units_configuration[] = {2, 3, 1};
    size_t units_configuration_len = sizeof(units_configuration) / sizeof(units_configuration[0]);

    enum Activation units_activation[] = {NN_SIGMOID, NN_SIGMOID, NN_SIGMOID, NN_SIGMOID, NN_SIGMOID,};

    NN * nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    /* Train */

    const float x_train[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };

    const float y_train[] = {
        0,
        1,
        1,
        0,
    };

    const size_t train_len = sizeof(x_train) / sizeof(x_train[0]) / units_configuration[0];

    const NN_train_opt opt = {
        .learning_rate = 0.5,
        .err_threshold = 0.0001,
        .err_epoch_logging = 1000,
        .mini_batch_size = train_len,
    };

    nn_fit(nn, x_train, y_train, train_len, &opt);

    /* Test */
    float out[1];

    for (size_t i = 0; i < 4; ++i) {
        nn_predict(nn, x_train + i*2, out);
        printf("%.0f XOR %.0f = [%f]\n", x_train[i*2], x_train[i*2 +1], *out);
    }

    nn_free(nn);
    return 0;
}
