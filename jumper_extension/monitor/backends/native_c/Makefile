CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
LDFLAGS ?= -lm -ldl
TARGET  := jumper_collector

SRC := collector.c
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all clean

all: $(DIR)$(TARGET)

$(DIR)$(TARGET): $(DIR)$(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(DIR)$(TARGET)
