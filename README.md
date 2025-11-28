# Neural network implementation

## Overview
Feedforward neural network implementation in plain and simple C99.

Model fitting uses **Backpropagation** for gradients computation and can use **Gradient Descent**, **Mini-batch GD** and **Stochastic GD** depending on the fit options (see ```batch_size``` attribute of ```NN_train_opt``` in ```include/neural_network.h```).

The loss can be computed using the **Mean Squared Error** and **Cross-entropy** (see ```loss_type``` attribute of ```NN_train_opt``` in ```include/neural_network.h```).

The current supported activation functions are **Sigmoid**, **RELU**, **Tanh** and **Softmax**.\
There are three weights initialization strategies, **Uniform**, **Glorot** and **He**.

The API is well explained in the ```include/neural_network.h``` header file.
There are some examples on how to use it in the ```examples``` directory.

## How to build
Just
``` bash
make
```
And you'll find all the examples executables in the ```build_examples``` directory.

## Requirements
- Any C99-compatible compiler.
- gnuplot, just used for examples chart view (the example works anyway, logging the train data into a file).
- wget and gunzip command line utils, just used in Makefile for download and uncompress MNIST dataset.

## MNIST example
The ```mnist.c``` example trains the neural network with the MNIST dataset.\
The results are not bad, train loss ~= 0.015 and test loss ~= 0.07 with this configuration:
```c
size_t units_configuration[] = {image_size, 128, 64, CLASS_NUM};
enum Activation units_activation[] = {NN_SIGMOID, NN_SIGMOID, NN_SOFTMAX};

const NN_train_opt opt = {
    .learning_rate = 0.1,
    .epochs = 150,
    .loss_log_fp = fp,
    .batch_size = 128,
    .loss_type = NN_CROSS_ENTROPY,
};
```

Obviously the training is much slow, because all the the computation uses only the cpu (almost 1 hour for 150 epochs):
```
./build_examples/mnist  3326,74s user 3,46s system 99% cpu 55:30,71 total
```

So I leave here the training chart if you don't want to wait:

![MNIST Loss chart](./mnist_train_pic.svg "MNIST Loss chart")

## TODO
- [ ] Implement all activations in CUDA (for feed_forward).
- [ ] Use CUDA functions in nn_fit and backprop.
