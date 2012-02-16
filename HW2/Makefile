#EXTRA_CFLAGS = -lrt -lpthread -O2 -lm
EXTRA_CFLAGS = -lrt -lpthread -ggdb -lm
#EXTRA_CFLAGS_CUDA = -lrt -arch=sm_13
EXTRA_CFLAGS_CUDA = -lrt -arch=sm_13 -g -G
CUDA_CC = /usr/local/cuda/bin/nvcc

info:
	echo "There is no 'default' target defined. Run make a3/a4/a5 to generate the a2/a3/a4"

all: clean a2 a4 a3

a3: p3.c util.c
	$(CC) -o $@ $^ $(EXTRA_CFLAGS)
	
a2: p2.c 
	$(CC) -o $@ $^ $(EXTRA_CFLAGS)

a4: p4.cu
	$(CUDA_CC) $(EXTRA_CFLAGS_CUDA) -o $@ $^
clean:
	rm -f a2 a3
