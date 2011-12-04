SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

OBJS = cg.o ${COMMON}/print_results.o  \
       ${COMMON}/${RAND}.o ${COMMON}/timers.o pmpi.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${FLINK} ${FLINKFLAGS} -o ${PROGRAM} ${OBJS} ${FMPI_LIB}

cg.o:		cg.f  mpinpb.h npbparams.h
	${FCOMPILE} cg.f pmpi.c

pmpi.o:		pmpi.c
	${CCOMPILE} pmpi.c -g

clean:
	- rm -f *.o *~ 
	- rm -f npbparams.h core



