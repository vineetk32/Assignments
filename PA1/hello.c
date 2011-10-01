#include <stdio.h>
#include "mpi.h"

int main(int argc,char **argv)
{
	int rank, retCode = 0,numProcessors;
	retCode = MPI_Init(&argc,&argv);
	if (retCode != MPI_SUCCESS)
	{
		printf("\nError during init! RetCode - %d\n",retCode);
		MPI_Abort(MPI_COMM_WORLD,retCode);
	}
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcessors);
	printf("\nRank %d:Hello, world!\n",rank);
	if (rank == 0)
	{
		printf("\nTotal processors spawned - %d",numProcessors);
	}
	MPI_Finalize();
	return 0;
}
