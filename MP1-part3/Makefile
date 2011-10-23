#---- Students, modify CC and CFLAGS as appropriate in your environment
CC = ~/Bin/gcc43
CFLAGS = -fopenmp

LIB = -lm -lc

UNITFILES = 

#------------ EXECUTABLE FILES : MOSTLY EXPERIMENTS ---------------

dlist : dlist.o $(UNITFILES)
	$(CC) $(CFLAGS) -o dlist dlist.o $(UNITFILES) $(LIB)

clean:
	rm -f core dlist *.o

dlist.o : dlist.c
	$(CC) $(CFLAGS) -c dlist.c $(INCLUDE) 


