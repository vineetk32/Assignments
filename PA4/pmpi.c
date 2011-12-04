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

	int i,size,rank,j;
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Status status;
	if (rank > 0)
	{
		PMPI_Send(myCallList,size,MPI_INT,0,rank,MPI_COMM_WORLD);
	}
	else if (rank == 0)
	{
		FILE *fout = fopen("matrix.data","w");
		if (fout == NULL)
		{
			printf("\nCouldnt open file!");
			MPI_Abort(MPI_COMM_WORLD,0);
		}

		for ( j = 0; j < size; j++)
		{
			fprintf(fout,"%d ",myCallList[j]);
		}
		for(i = 1; i < size; i++)
		{
			for ( j = 0; j < size; j++)
				myCallList[j] = 0;

			PMPI_Recv(myCallList,size,MPI_INT,i,i,MPI_COMM_WORLD,&status);

			fprintf(fout,"\n");
			for ( j = 0; j < size; j++)
				fprintf(fout,"%d ",myCallList[j]);
		}
		fclose(fout);
			
	}
	return PMPI_Finalize();
}
