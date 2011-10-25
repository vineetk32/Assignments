#---- Students, modify CC and CFLAGS as appropriate in your environment
CFLAGS = -fopenmp -O2

LIB = -lm -lc -lrt

UNITFILES = 

#------------ EXECUTABLE FILES : MOSTLY EXPERIMENTS ---------------

all: clean dlist dlist_coarse dlist_fine

dlist : dlist.o $(UNITFILES)
	$(CC) $(CFLAGS) -o dlist dlist.o $(UNITFILES) $(LIB)

dlist_coarse : dlist_coarse.o $(UNITFILES)
	$(CC) $(CFLAGS) -o dlist_coarse dlist_coarse.o $(UNITFILES) $(LIB)

dlist_fine : dlist_fine.o $(UNITFILES)
	$(CC) $(CFLAGS) -o dlist_fine dlist_fine.o $(UNITFILES) $(LIB)

clean:
	rm -f core dlist dlist_coarse dlist_coarse_kedar dlist_fine *.o

dlist.o : dlist.c
	$(CC) $(CFLAGS) -c dlist.c  $(INCLUDE) 

dlist_coarse.o : dlist_coarse.c
	$(CC) $(CFLAGS) -c dlist_coarse.c  $(INCLUDE) 

dlist_fine.o : dlist_fine.c
	$(CC) $(CFLAGS) -c dlist_fine.c  $(INCLUDE) 
