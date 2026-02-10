CC = mpicc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp
TARGET = mitm

all: $(TARGET)

$(TARGET): mitm.c
	$(CC) $(CFLAGS) -o $(TARGET) mitm.c $(LDFLAGS)

clean:
	rm -f $(TARGET)