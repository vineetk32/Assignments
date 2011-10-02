#---- Students, modify CC and CFLAGS as appropriate in your environment
CC = gcc
CFLAGS = -fopenmp -m32

LIB = -lm -lc

UNITFILES = 

#------------ EXECUTABLE FILES : MOSTLY EXPERIMENTS ---------------

gauss : gauss.o $(UNITFILES)
	$(CC) $(CFLAGS) -o gauss gauss.o $(UNITFILES) $(LIB)

clean:
	rm -f core dlist *.o

gauss.o : gauss.c
	$(CC) $(CFLAGS) -c gauss.c $(INCLUDE) 


