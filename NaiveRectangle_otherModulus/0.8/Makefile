CC = gcc
CFLAGS = -std=c99 -Wall -Werror -pedantic-errors -lm -fopenmp -O3

# Source files and targets
SOURCES = ux.c vy.c modulus.c
TARGETS = ux vy modulus

# Default target
all: $(TARGETS)

# Pattern rule for building targets
$(TARGETS): %: %.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

