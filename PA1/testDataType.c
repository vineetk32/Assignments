#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

struct sMatrix
{
	double **matrix;
	int rows;
	int cols;
};

void allocMatrixStruct(struct sMatrix *matrix,int rows,int cols)
{
	int i;
	double *data;
	matrix->cols = cols;
	matrix->rows = rows;
	data = (double *) malloc(sizeof(double) * rows * cols);
	matrix->matrix = (double **) malloc (sizeof(double *) * rows);
	for (i = 0 ; i < rows; i++)
	{
		matrix->matrix[i] = &(data[cols*i]);
	}
}

int main(int argc,char **argv)
{
	int rank, retCode = 0,numProcessors,i,j;
	retCode = MPI_Init(&argc,&argv);
	struct sMatrix testmatrix;
	double **tempMatrix;
	double *tempData;

	tempData = (double *) malloc(sizeof(double) * 3 * 3);
	tempMatrix = (double **) malloc (sizeof(double *) * 3);
	for (i = 0 ; i < 3; i++)
	{
		tempMatrix[i] = &(tempData[3*i]);
	}

	MPI_Status status;
	allocMatrixStruct(&testmatrix,3,3);

	for ( i = 0; i < 3 ; i++)
	{
		for (j = 0; j < 3; j++)
		{
			testmatrix.matrix[i][j] = 1;
		}
	}

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

	/*MPI_Aint base;
	MPI_Aint matrixRowAddresses[3];
	MPI_Datatype rowType;
	MPI_Address(&testmatrix,&base);
	for ( i = 0; i < 3 ; i++)
	{
		MPI_Address(&(testmatrix.matrix[i]),&(matrixRowAddresses[i][j]));
		matrixRowAddresses[i] -= base;
	}
	MPI_Type_struct(3,3,*/

	if (rank == 0)
	{
		MPI_Send(&testmatrix.matrix[0][0],3*3,MPI_DOUBLE,1,1,MPI_COMM_WORLD);
		MPI_Send(&testmatrix.rows,1,MPI_INTEGER,1,2,MPI_COMM_WORLD);
		MPI_Send(&testmatrix.cols,1,MPI_INTEGER,1,3,MPI_COMM_WORLD);
		/*for ( i = 0; i < testmatrix.rows; i++)
		{
			printf("\n\t\t\t");
			for (j = 0; j < testmatrix.cols; j++)
			{
				printf("\t%d:%lf",rank,testmatrix.matrix[i][j]);
			}
			
		}*/
		printf("\n");

	}
	else if (rank == 1)
	{
		int rows,cols,retCode;

		retCode = MPI_Recv(&tempMatrix[0][0],3*3,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status);
		if (retCode == MPI_SUCCESS)
		{
			retCode = MPI_SUCCESS;
			retCode = MPI_Recv(&rows,1,MPI_INTEGER,0,2,MPI_COMM_WORLD,&status);
			if ( retCode == MPI_SUCCESS)
			{
				retCode = MPI_SUCCESS;
				retCode = MPI_Recv(&cols,1,MPI_INTEGER,0,3,MPI_COMM_WORLD,&status);
				if (retCode == MPI_SUCCESS)
				{
					for ( i = 0; i < rows; i++)
					{
						printf("\n");
						for (j = 0; j < cols; j++)
						{
							printf("\t%d:%lf",rank,tempMatrix[i][j]);
						}
					}
					printf("\n");
				}
				else
				{
					printf("\nMPI_Recv cols got %d",retCode);
				}
			}
			else
			{
				printf("\nMPI_Recv rows got %d",retCode);
			}
		}
		else
		{
			printf("\nMPI_Recv tempMat got %d",retCode);
		}
	}
	MPI_Finalize();
	return 0;
}
