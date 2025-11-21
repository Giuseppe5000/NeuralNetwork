#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../neural_network.h"

FILE *popen(const char *command, const char *type);
int pclose(FILE *stream);

#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])
#define CLASS_NUM 10

/*
* Transoform the input number (in big endian) into a little endian number.
*/
uint32_t to_little_endian(uint32_t big_endian_num) {
    uint8_t b1, b2, b3, b4;

    b1 = big_endian_num & 255;
    b2 = (big_endian_num >> 8) & 255;
    b3 = (big_endian_num >> 16) & 255;
    b4 = (big_endian_num >> 24) & 255;

    return ((uint32_t)b1 << 24) + ((uint32_t)b2 << 16) + ((uint32_t)b3 << 8) + b4;
}

/*
* Read the image data from the IDX file returning the pointer to the float data.
* It also fill 'img_len' with the number of images and 'img_size' with the byte size of one image.
*
* (https://github.com/cvdfoundation/mnist?tab=readme-ov-file#file-format)
* IDX format is a simple format for vectors and multidimensional matrices of various numerical types.
* The basic format is:
* ---------------------------
*  magic number
*  size in dimension 0
*  size in dimension 1
*  size in dimension 2
*  .....
*  size in dimension N
*  data
* ---------------------------
*
* The magic number is an integer (big endian / MSB).
* The first 2 bytes are always 0.
* The third byte codes the type of the data:
* ---------------------------
*  0x08: unsigned byte
*  0x09: signed byte
*  0x0B: short (2 bytes)
*  0x0C: int (4 bytes)
*  0x0D: float (4 bytes)
*  0x0E: double (8 bytes)
* ---------------------------
* The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices...
*/
float *read_mnist_images(const char *path, size_t *img_len, size_t *img_size) {
    FILE *fp = fopen(path, "rb");

    if (fp == NULL) {
        fprintf(stderr, "[ERROR]: Cannot open %s\n", path);
        exit(1);
    }

    /* Magic number */
    uint8_t magic_number[4] = {0};
    size_t items_read = fread(magic_number, sizeof(uint8_t), 4, fp);

    if (items_read != 4) {
        fprintf(stderr, "[ERROR]: Expected 4 bytes!\n");
        exit(1);
    }

    if (magic_number[0] != 0 || magic_number[1] != 0) {
        fprintf(stderr, "[ERROR]: Expected 0 as the first two bytes of magic number!\n");
        exit(1);
    }

    if (magic_number[2] != 8) {
        fprintf(stderr, "[ERROR]: Expected unsigned byte (0x08) data type!\n");
        exit(1);
    }

    if (magic_number[3] != 3) {
        fprintf(stderr, "[ERROR]: Expected 3 dimensions!\n");
        exit(1);
    }

    /* Dimensions */
    uint32_t dimensions[3] = {0};
    items_read = fread(dimensions, sizeof(uint32_t), 3, fp);

    if (items_read != 3) {
        fprintf(stderr, "[ERROR]: Expected 3 dimensions!\n");
        exit(1);
    }

    for (size_t i = 0; i < 3; ++i) {
        dimensions[i] = to_little_endian(dimensions[i]);
    }

    *img_len = dimensions[0];
    *img_size = dimensions[1] * dimensions[2];

    /* Data */
    const size_t data_size = (*img_size) * (*img_len);
    uint8_t *data = malloc(sizeof(uint8_t) * data_size);
    items_read = fread(data, sizeof(uint8_t), data_size, fp);

    if (items_read != data_size) {
        fprintf(stderr, "[ERROR]: Expected %zu bytes!\n", data_size);
        exit(1);
    }

    fclose(fp);

    /* Convert data to float (range [0..1]) */
    float *data_float = malloc(sizeof(float) * data_size);

    for (size_t i = 0; i < data_size; ++i) {
        data_float[i] = (float) data[i] / 255.0;
    }

    free(data);
    return data_float;
}

/*
* Like 'read_mnist_images' but for the labels.
*/
float *read_mnist_labels(const char *path) {
    FILE *fp = fopen(path, "rb");

    if (fp == NULL) {
        fprintf(stderr, "[ERROR]: Cannot open %s\n", path);
        exit(1);
    }

    /* Magic number */
    uint8_t magic_number[4] = {0};
    size_t items_read = fread(magic_number, sizeof(uint8_t), 4, fp);

    if (items_read != 4) {
        fprintf(stderr, "[ERROR]: Expected 4 bytes!\n");
        exit(1);
    }

    if (magic_number[0] != 0 || magic_number[1] != 0) {
        fprintf(stderr, "[ERROR]: Expected 0 as the first two bytes of magic number!\n");
        exit(1);
    }

    if (magic_number[2] != 8) {
        fprintf(stderr, "[ERROR]: Expected unsigned byte (0x08) data type!\n");
        exit(1);
    }

    if (magic_number[3] != 1) {
        fprintf(stderr, "[ERROR]: Expected 1 dimension!\n");
        exit(1);
    }

    /* Dimensions */
    uint32_t dimension = 0;
    items_read = fread(&dimension, sizeof(uint32_t), 1, fp);

    if (items_read != 1) {
        fprintf(stderr, "[ERROR]: Expected 1 dimension!\n");
        exit(1);
    }

    dimension = to_little_endian(dimension);

    /* Data */
    uint8_t *data = malloc(sizeof(uint8_t) * dimension);
    items_read = fread(data, sizeof(uint8_t), dimension, fp);

    if (items_read != dimension) {
        fprintf(stderr, "[ERROR]: Expected %u bytes!\n", dimension);
        exit(1);
    }

    fclose(fp);

    /*
    * Each element of data is a number in [0,9],
    * but the train need it in one-hot format.
    */
    float *data_one_hot = calloc(dimension * CLASS_NUM, sizeof(float));

    for (size_t i = 0; i < dimension; ++i) {
        uint8_t label = data[i];
        data_one_hot[i*CLASS_NUM + label] = 1.0;
    }

    free(data);

    return data_one_hot;
}

int main(void) {
    /* Reading training and test data */
    const char *images_file_path = "train-images-idx3-ubyte";
    const char *labels_file_path = "train-labels-idx1-ubyte";
    const char *images_test_file_path = "t10k-images-idx3-ubyte";
    const char *labels_test_file_path = "t10k-labels-idx1-ubyte";

    size_t train_imgs_len = 0;
    size_t image_size = 0;
    float *train_imgs = read_mnist_images(images_file_path, &train_imgs_len, &image_size);
    float *train_labels = read_mnist_labels(labels_file_path);

    size_t test_imgs_len = 0;
    float *test_imgs = read_mnist_images(images_test_file_path, &test_imgs_len, &image_size);
    float *test_labels = read_mnist_labels(labels_test_file_path);

    /* Network init */
    size_t units_configuration[] = {image_size, 128, 64, CLASS_NUM};
    enum Activation units_activation[] = {NN_SIGMOID, NN_SIGMOID, NN_SOFTMAX};
    size_t units_configuration_len = ARRAY_LEN(units_configuration);

    NN *nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    /* Train */
    FILE *fp = fopen("mnist_loss.txt", "w");
    const NN_train_opt opt = {
        .learning_rate = 0.1,
        .epochs = 150,
        .loss_log_fp = fp,
        .batch_size = 128,
        .loss_type = NN_CROSS_ENTROPY,
    };

    nn_fit(nn, train_imgs, train_labels, train_imgs_len, test_imgs, test_labels, test_imgs_len, &opt);

    /* Memory free */
    fclose(fp);
    free(train_imgs);
    free(train_labels);
    free(test_imgs);
    free(test_labels);
    nn_free(nn);

    /* Plotting */
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "set grid \n");
    fprintf(gnuplotPipe, "set xlabel \"Epochs\" \n");
    fprintf(gnuplotPipe, "set ylabel \"Loss\" \n");
    fprintf(gnuplotPipe, "plot '%s' using 1:2 with lines title 'Train loss' lc rgb \"blue\", ", "mnist_loss.txt");
    fprintf(gnuplotPipe, "'%s' using 1:3 with lines title 'Test loss' lc rgb \"red\" \n", "mnist_loss.txt");
    pclose(gnuplotPipe);

    return 0;
}
