#---- Students, modify CC and CFLAGS as appropriate in your environment
CC = gcc
CFLAGS = -O2

LIB = -lm -lc -lrt

UNITFILES = 

#------------ EXECUTABLE FILES : MOSTLY EXPERIMENTS ---------------

all: clean gauss gauss_seq gauss_j gauss_i

gauss : gauss.o $(UNITFILES)
	$(CC) $(CFLAGS) -fopenmp -o gauss gauss.o $(UNITFILES) $(LIB)

gauss_seq : gauss_seq.o 
	$(CC) $(CFLAGS) -o gauss_seq gauss_seq.o $(UNITFILES) $(LIB)
gauss_j : gauss_j.o 
	$(CC) $(CFLAGS) -o gauss_j gauss_j.o $(UNITFILES) $(LIB)
gauss_i : gauss_i.o 
	$(CC) $(CFLAGS) -o gauss_i gauss_i.o $(UNITFILES) $(LIB)
clean:
	rm -f core dlist *.o gauss gauss_seq

gauss.o : gauss.c
	$(CC) $(CFLAGS) -c gauss.c $(INCLUDE) 

gauss_seq.o : gauss_seq.c
	$(CC) $(CFLAGS) -c gauss_seq.c $(INCLUDE) 
gauss_j.o : gauss_j.c
	$(CC) $(CFLAGS) -c gauss_j.c $(INCLUDE) 
gauss_i.o : gauss_i.c
	$(CC) $(CFLAGS) -c gauss_i.c $(INCLUDE) 
