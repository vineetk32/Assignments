CC=gcc
FLAGS=-lrt -ggdb
OBJDIR=objs
BINDIR=bin
#COMPILE_FLAGS = -D__DEBUG
COMPILE_FLAGS =

OBJS = logStats.o util.o

%.o:	%.c
	${CC} -c -o ${OBJDIR}/$@ $< -ggdb $(COMPILE_FLAGS)

mt:	util.o logStats_t.o
	${CC} -o ${BINDIR}/logStats_t.x ${OBJDIR}/logStats_t.o ${OBJDIR}/util.o ${FLAGS} -lpthread

nmt:	util.o logStats.o
	${CC} -o ${BINDIR}/logStats.x ${OBJDIR}/logStats.o ${FLAGS} 

clean:
	rm ${BINDIR}/*.x ${OBJDIR}/*.o
