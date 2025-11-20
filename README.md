# Neural network implementation

## Overview
Feedforward neural network implementation in plain and simple C99.

Model fitting uses **Backpropagation** for gradients computation and can use **Gradient Descent**, **Mini-batch GD** and **Stochastic GD** depending on the fit options (see ```batch_size``` attribute of ```NN_train_opt``` in ```neural_network.h```).

The loss can be computed using the **Mean Squared Error** and **Cross-entropy** (see ```loss``` attribute of ```NN_train_opt``` in ```neural_network.h```).

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

## TODO
- [ ] Maybe it is better to have only one single file with train and test loss.
- [ ] There are some warnings regarding fread.
- [ ] Add the image of mnist loss in the readme.
- [ ] Tell in the readme that the makefile expects that wget and gunzip are installed.
