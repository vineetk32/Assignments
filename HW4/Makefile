#EXTRA_CFLAGS = -lrt -lpthread -O2 -lm
EXTRA_CFLAGS = -lrt -lpthread -ggdb -lm

info:
	echo "There is no 'default' target defined. Run make a3 to generate the a3"

all: clean a3

a3: p3.c 
	$(CC) -o $@ $^ $(EXTRA_CFLAGS)
	
clean:
	rm -f a3
