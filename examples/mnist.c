#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../neural_network.h"

FILE *popen(const char *command, const char *type);
int pclose(FILE *stream);

#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])

/*
* Transoform the input number (in big endian) into a little endian number.
*/
int32_t to_little_endian(int32_t big_endian_num) {
    unsigned char b1, b2, b3, b4;

    b1 = big_endian_num & 255;
    b2 = (big_endian_num >> 8) & 255;
    b3 = (big_endian_num >> 16) & 255;
    b4 = (big_endian_num >> 24) & 255;

    return ((int32_t)b1 << 24) + ((int32_t)b2 << 16) + ((int32_t)b3 << 8) + b4;
}

/*
* Read the image data from the IDX file returning the pointer to the float data.
* It also fill 'img_len' with the number of images and 'img_size' with the byte size of one image.
*
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
    uint32_t magic_number = 0;
    fread(&magic_number, sizeof(uint32_t), 1, fp);
    const uint8_t *magic_num_bytes = (uint8_t*) &magic_number;

    printf("Magic number:\n");
    printf("First byte: 0x%02X\n", magic_num_bytes[0]);
    printf("Second byte: 0x%02X\n", magic_num_bytes[1]);
    printf("Third byte: 0x%02X\n", magic_num_bytes[2]);
    printf("Fourth byte: 0x%02X\n\n", magic_num_bytes[3]);

    if (magic_num_bytes[2] != 0x08) {
        fprintf(stderr, "[ERROR]: Expected unsigned byte (0x08) data type!\n");
        exit(1);
    }

    if (magic_num_bytes[3] != 0x03) {
        fprintf(stderr, "[ERROR]: Expected 3 dimensions!\n");
        exit(1);
    }

    /* Dimensions */
    uint32_t dimensions[3] = {0};
    fread(dimensions, sizeof(uint32_t), 3, fp);

    for (size_t i = 0; i < 3; ++i) {
        dimensions[i] = to_little_endian(dimensions[i]);
    }

    *img_len = dimensions[0];
    *img_size = dimensions[1] * dimensions[2];

    /* Data */
    const size_t data_size = (*img_size) * (*img_len);
    uint8_t *data = malloc(sizeof(uint8_t) * data_size);
    fread(data, sizeof(uint8_t), data_size, fp);
    fclose(fp);

    /* Convert data to float (range [0..1]) */
    float *data_float = malloc(sizeof(float) * data_size);

    for (size_t i = 0; i < data_size; ++i) {
        data_float[i] = (float) data[i] / 255.0;
        printf("data[%zu] = %f\n", i, data_float[i]);
    }

    free(data);
    return data_float;
}

int main(void) {
    const char* images_file_path = "train-images-idx3-ubyte";
    const char* labels_file_path = "train-labels-idx1-ubyte";

    /* Reading training data */
    size_t train_imgs_len = 0;
    size_t image_size = 0;
    float *train_imgs = read_mnist_images(images_file_path, &train_imgs_len, &image_size);

    size_t train_labels_len = 0;
    // float *train_labels = read_mnist_labels(labels_file_path, &train_labels_len);

    /* Network init */
    // size_t units_configuration[] = {image_size, ............ 10};
    // enum Activation units_activation[] =
    // size_t units_configuration_len = ARRAY_LEN(units_configuration);

    // NN * nn = nn_init(units_configuration, units_configuration_len, units_activation, NN_GLOROT);

    // FILE* fp = fopen("mnist_train.txt", "w");

    // const NN_train_opt opt = {
    //     .learning_rate = ,
    //     .epoch_num = ,
    //     .log_fp = fp,
    //     .batch_size = ,
    //     .loss = ,
    // };

    /* Train */
    // nn_fit(nn, train_imgs, train_labels, train_imgs_len, &opt);

    /* Memory free */
    // nn_free(nn);
    // fclose(fp);
    free(train_imgs);
    // free(train_labels);

    /* Plotting */
    // FILE* gnuplotPipe = popen("gnuplot -persistent", "w");
    // fprintf(gnuplotPipe, "set grid \n");
    // fprintf(gnuplotPipe, "set xlabel \"Epochs\" \n");
    // fprintf(gnuplotPipe, "set ylabel \"Loss\" \n");
    // fprintf(gnuplotPipe, "plot '%s' lc rgb \"black\" \n", "mnist_train.txt");
    // pclose(gnuplotPipe);

    return 0;
}
