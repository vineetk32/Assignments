CC=gcc
FLAGS=-lrt
OBJDIR=objs
BINDIR=bin

OBJS = logStats.o

%.o:	%.c
	${CC} -c -o ${OBJDIR}/$@ $< 

all:	${OBJS}
	${CC} -o ${BINDIR}/logStats.x ${OBJDIR}/*.o ${FLAGS} 

clean:
	rm ${BINDIR}/*.x ${OBJDIR}/*.o
