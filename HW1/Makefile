CC=gcc
FLAGS=-lrt -ggdb
OBJDIR=objs
BINDIR=bin

OBJS = logStats.o util.o

%.o:	%.c
	${CC} -c -o ${OBJDIR}/$@ $< -ggdb

mt:	util.o logStats_t.o
	${CC} -o ${BINDIR}/logStats_t.x ${OBJDIR}/logStats_t.o ${OBJDIR}/util.o ${FLAGS} -lpthread

nmt:	util.o logStats.o
	${CC} -o ${BINDIR}/logStats.x ${OBJDIR}/logStats.o ${FLAGS} 

clean:
	rm ${BINDIR}/*.x ${OBJDIR}/*.o
