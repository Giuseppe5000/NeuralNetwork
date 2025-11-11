# Neural network implementation

## Overview
Feedforward neural network implementation in plain and simple C99.

Model fitting uses **Backpropagation** for gradients computation and can use **Gradient Descent**, **Mini-batch GD** and **Stochastic GD** depending on the fit options (see ```batch_size``` attribute of ```NN_train_opt``` in ```neural_network.h```).\
The error is calculated using the Mean Squared Error.

The current supported activation functions are **Sigmoid**, **RELU**, **Tanh** and **Softmax**.\
There are three weights initialization strategies, **Uniform**, **Glorot** and **He**.

The API is well explained in the ```neural_network.h``` header file.
There are some examples on how to use it in the ```examples``` directory.

## How to build
Just
``` bash
make
```
And you'll find all the examples executables in the ```build_examples``` directory.

## Requirements
- Any C99-compatible compiler.
- Gnuplot, only needed for examples chart view (the example works anyway, logging the train data into a file).

## TODOs
- [x] Optimize the use of intermediate_products in feed\_forward (by removing the res var). But in this way should store the activations and not the products. (Btw intermediate\_products can be stored on the NN struct).\ Maybe i can store on NN both intermediate\_products and intermediate\_activations.
- [x] Remove the use of VLAs from backprop, by using an allocated big enough space with malloc at the start of fit.
- [x] Fix mini batch implementation by taking random n elements each time (at the moment takes always the first n elements, so it is wrong).
- [x] Add Softmax activation function (only final layer).
- [x] It is possible to eliminate intermediate_products and use the intermediate activations to compute the derivatives of products, this is possible because the derivatives (for now) are defined in terms of the primitive function. Before doing this it is better to implement softmax.
- [x] Check const correctness.
- [x] Add train option for chioce between MSE and Cross-entropy loss.
- [x] Improve comments and documentation.
- [ ] Potential nan when computing cross entropy (check this even for weights init).
- [ ] Simplify the code where is possible.
