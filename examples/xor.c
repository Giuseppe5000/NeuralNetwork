#include <stdio.h>
#include "../neural_network.h"

FILE *popen(const char *command, const char *type);
int pclose(FILE *stream);

#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])

int main(void) {
    size_t units_configuration[] = {2, 2, 1};
    enum Activation units_activation[] = {NN_SIGMOID, NN_SIGMOID};
    size_t units_configuration_len = ARRAY_LEN(units_configuration);

    NN * nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    /* Train data */
    const float x_train[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    };
    const size_t train_len = ARRAY_LEN(x_train) / units_configuration[0];

    const float y_train[] = {
        0,
        1,
        1,
        0,
    };

    const char *file_path = "xor_train.txt";
    FILE* fp = fopen(file_path, "w");
    if (fp == NULL) {
        fprintf(stderr, "[ERROR]: Cannot open file %s\n", file_path);
        return 1;
    }

    const NN_train_opt opt = {
        .learning_rate = 2,
        .epoch_num = 1500,
        .log_fp = fp,
        .batch_size = 4,
    };

    nn_fit(nn, x_train, y_train, train_len, &opt);

    /* Test */
    float out[1];

    for (size_t i = 0; i < 4; ++i) {
        nn_predict(nn, x_train + i*2, out);
        printf("%.0f XOR %.0f = [%f]\n", x_train[i*2], x_train[i*2 +1], *out);
    }

    nn_free(nn);
    fclose(fp);

    /* Plotting */
    FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "set grid \n");
    fprintf(gnuplotPipe, "set xlabel \"Epochs\" \n");
    fprintf(gnuplotPipe, "set ylabel \"Loss\" \n");
    fprintf(gnuplotPipe, "plot '%s' lc rgb \"black\" \n", "xor_train.txt");
    pclose(gnuplotPipe);

    return 0;
}
