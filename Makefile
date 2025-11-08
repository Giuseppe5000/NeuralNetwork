CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99 -Werror=vla 
# For more performance add: -Ofast -march=native -ffast-math
LIBS= -lm
SRC_DIR = examples
BUILD_DIR = build_examples
SOURCES = $(wildcard $(SRC_DIR)/*.c)
EXECUTABLES = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%,$(SOURCES))

all: $(BUILD_DIR) $(EXECUTABLES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) neural_network.c $< -o $@ $(LIBS) $(CFLAGS)

clean:
	rm -rf $(BUILD_DIR)
