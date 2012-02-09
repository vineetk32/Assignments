EXTRA_CFLAGS = -lrt -lpthread
EXTRA_CFLAGS_CUDA = -lrt -arch=sm_13
CUDA_CC = /usr/local/cuda/bin/nvcc

all: a2 a4

a2: p2.c 
	$(CC) -o $@ $^ -lpthread -lrt

a4: p4.cu
	$(CUDA_CC) $(EXTRA_CFLAGS_CUDA) -o $@ $^