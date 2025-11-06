# Neural network implementation

## Overview
Feedforward neural network implementation in plain and simple C99.

Model fitting uses **Backpropagation** for gradients computation and can use **Gradient Descent**, **Mini-batch GD** and **Stochastic GD** depending on the fit options (see ```mini_batch_size``` attribute of ```NN_train_opt``` in ```neural_network.h```).\
The error is calculated using the Mean Squared Error.

The current supported activation functions are **Sigmoid**, **RELU** and **Tanh**.\
There are three weights initialization strategies, **Uniform**, **Glorot** and **He**.

The API is well explained in the ```neural_network.h``` header file.
There are some examples on how to use it in the ```examples``` directory.

## How to build
Just
``` bash
make
```
And you'll find all the examples executables in the ```build_examples``` directory.

## TODOs
- [ ] Fix mini batch implementation by taking random n elements each time (at the moment takes always the first n elements, so it is wrong).
- [x] VLAs can cause stack overflow, so it is better to use nn_malloc and free for the big arrays.
- [ ] Simplify the code where is possible.
- [ ] Check const correctness
- [ ] Improve comments and documentation.
