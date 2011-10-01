## Please use 'add intel_mpich2_hydra-101' before compiling

CC=mpicc
CFLAGS=-O2 -c99
#CFLAGS=-g -static -Wall
LIB=-lmpich -lm
SRC=\
    matrixMult_mpi.c \

OBJ=$(SRC:.c=.o)

OBJDIR=.
BINDIR=.


.c.o:	
	$(CC) -c $(CFLAGS) $<

matrixMult_mpi:  $(OBJ) $(ENVFILE)
	$(CC) $(CFLAGS) -o ${BINDIR}/$@.x $(OBJ) $(LIB) 

clean:
	rm -f *.o  ${BINDIR}/*.x

run:
	bsub -W 00:15 < jobScript	# You must edit my_mm.script

