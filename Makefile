CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99 -Werror=vla -O3 -march=native -ffast-math
LIBS= -lm
SRC_DIR = examples
BUILD_DIR = build_examples
SOURCES = $(wildcard $(SRC_DIR)/*.c)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%,$(SOURCES))

all: $(BUILD_DIR) $(EXECUTABLES) mnist_dataset

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) neural_network.c $< -o $@ $(LIBS) $(CFLAGS)

mnist_dataset: train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

train-images-idx3-ubyte:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	gunzip -d train-images-idx3-ubyte.gz

train-labels-idx1-ubyte:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
	gunzip -d train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
	gunzip -d t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte:
	wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
	gunzip -d t10k-labels-idx1-ubyte.gz

clean:
	rm -rf $(BUILD_DIR)
	rm *-ubyte
	rm *.txt
