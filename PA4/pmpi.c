#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

extern int MPI_Init(int *argc,char ***argv) __attribute__((weak));
extern int MPI_Finalize(void) __attribute__((weak));

int *myCallList;
int MPI_Init(int *argc,char ***argv)
{
	int size,rank,retVal;
	retVal = PMPI_Init(argc,argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	if (size > 0)
	{
		myCallList = (int *) malloc(sizeof(int) * size);
	}
	return retVal;
}

int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,MPI_Comm comm)
{
	myCallList[dest]++;
	return PMPI_Send(buf,count,datatype,dest,tag,comm);
}

int MPI_Finalize(void)
{

	int i,size,rank;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	printf("%d:myCallList - ",rank);
	for (i = 0; i < size; i++)
	{
		printf("%d\t",myCallList[i]);	
	}
	printf("\n");
	return PMPI_Finalize();
}
