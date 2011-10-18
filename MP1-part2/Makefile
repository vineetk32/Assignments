#----- Students, modify CC and CFLAGS as appropriate in your environment
CC = ~/Bin/gcc43
CFLAGS = -fopenmp

LIB = -lm -lc

UNITFILES = 

#------------ EXECUTABLE FILES : MOSTLY EXPERIMENTS ---------------

histogram : histogram.o $(UNITFILES)
	$(CC) $(CFLAGS) -o histogram histogram.o $(UNITFILES) $(LIB)

clean:
	rm -f core histogram *.o

histogram.o : histogram.c
	$(CC) $(CFLAGS) -c histogram.c $(INCLUDE) 


