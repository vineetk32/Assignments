#EXTRA_CFLAGS = -lrt -lpthread -O2 -lm
EXTRA_CFLAGS = -lrt -lpthread -ggdb -lm

info:
	echo "There is no 'default' target defined. Run make a3/a4/a5 to generate the a2/a3/a4"

all: clean a2 a4 a3

a3: p3.c 
	$(CC) -o $@ $^ $(EXTRA_CFLAGS)
	
a4: p4.c
	$(CC) $(EXTRA_CFLAGS) -o $@ $^
clean:
	rm -f a3 a4
