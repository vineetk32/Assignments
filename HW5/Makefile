#EXTRA_CFLAGS = -lrt -lpthread -O2 -lm
EXTRA_CFLAGS = -lrt -lpthread -ggdb -lm -Wall

info:
	echo "There is no 'default' target defined. Run make a3 to generate the a3"

all: clean a3

a3: p3.c util.c
	gcc -o $@ $^ -g -lrt
	
clean:
	rm -f a3
