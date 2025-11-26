CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99 -Werror=vla -O3 -march=native -ffast-math
LIBS= -lm -L/lib/cuda/lib64/ -lcudart -lcublas
SRC_DIR = examples
BUILD_DIR = build_examples
SOURCES = $(wildcard $(SRC_DIR)/*.c)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%,$(SOURCES))

all: $(BUILD_DIR) $(EXECUTABLES) mnist_dataset

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.c nn_cuda.o | $(BUILD_DIR)
	$(CC) src/neural_network.c $(BUILD_DIR)/nn_cuda.o $< -o $@ $(LIBS) $(CFLAGS)

nn_cuda.o: src/nn_cuda.cu
	nvcc -c src/nn_cuda.cu -o $(BUILD_DIR)/nn_cuda.o

mnist_dataset: $(BUILD_DIR)/train-images-idx3-ubyte $(BUILD_DIR)/train-labels-idx1-ubyte $(BUILD_DIR)/t10k-images-idx3-ubyte $(BUILD_DIR)/t10k-labels-idx1-ubyte

$(BUILD_DIR)/train-images-idx3-ubyte:
	wget -P $(BUILD_DIR) https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
	gunzip -dc $(BUILD_DIR)/train-images-idx3-ubyte.gz > $(BUILD_DIR)/train-images-idx3-ubyte
	rm $(BUILD_DIR)/train-images-idx3-ubyte.gz

$(BUILD_DIR)/train-labels-idx1-ubyte:
	wget -P $(BUILD_DIR) https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
	gunzip -dc $(BUILD_DIR)/train-labels-idx1-ubyte.gz > $(BUILD_DIR)/train-labels-idx1-ubyte
	rm $(BUILD_DIR)/train-labels-idx1-ubyte.gz

$(BUILD_DIR)/t10k-images-idx3-ubyte:
	wget -P $(BUILD_DIR) https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
	gunzip -dc $(BUILD_DIR)/t10k-images-idx3-ubyte.gz > $(BUILD_DIR)/t10k-images-idx3-ubyte
	rm $(BUILD_DIR)/t10k-images-idx3-ubyte.gz

$(BUILD_DIR)/t10k-labels-idx1-ubyte:
	wget -P $(BUILD_DIR) https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
	gunzip -dc $(BUILD_DIR)/t10k-labels-idx1-ubyte.gz > $(BUILD_DIR)/t10k-labels-idx1-ubyte
	rm $(BUILD_DIR)/t10k-labels-idx1-ubyte.gz

clean:
	rm -rf $(BUILD_DIR)
	rm *.txt
