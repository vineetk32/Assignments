CC=gcc
#FLAGS=-lrt -ggdb
FLAGS=-lrt
OBJDIR=objs
BINDIR=bin
#COMPILE_FLAGS = -D__DEBUG
COMPILE_FLAGS =

OBJS = p3.o util.o

info:
	echo "There is no 'all' target defined. Run make a3/a4/a5 to generate the a3.x/a4.x/libmythread.a"

%.o:	%.c
	${CC} -c -o $@ $< -ggdb $(COMPILE_FLAGS)

a3:	util.o p3.o
	${CC} -o p3.x p3.o util.o ${FLAGS} -lpthread -lrt

a4:
	nvcc p4.cu -o p4.x

clean:
	rm *.x *.o
