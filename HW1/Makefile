CC=gcc
FLAGS=-lrt
OBJDIR=objs
BINDIR=bin

OBJS = logStats.o

%.o:	%.c
	${CC} -c -o ${OBJDIR}/$@ $< 

mt:	logStats_t.o
	${CC} -o ${BINDIR}/logStats_t.x ${OBJDIR}/logStats_t.o ${FLAGS} 

nmt:	logStats.o
	${CC} -o ${BINDIR}/logStats.x ${OBJDIR}/logStats.o ${FLAGS} 

clean:
	rm ${BINDIR}/*.x ${OBJDIR}/*.o
