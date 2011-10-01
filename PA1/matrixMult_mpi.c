#include "matrixMult_mpi.h"

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
		// So that all the matrix elements are in contiguous memory locations.
		matrix->matrix[i] = &(data[cols*i]);
	}
}


void freeMatrixStruct(struct sMatrix *matrix)
{
	int i;
	matrix->cols = -1;
	matrix->rows = -1;
	for (i = 0 ; i < matrix->rows; i++)
	{
		free(matrix->matrix[i]);
	}
	free(matrix->matrix);
}

int getNumPartialResults(int *powerBuffer)
{
	int count = 0,i;
	for (i = 0;i < 10;i++)
	{
		if (powerBuffer[i] == -1)
		{
			return count;
		}
		else if (powerBuffer[i] != 0)
		{
			count++;
		}
	}
	return count;
}
int getLargestPartialResult(int *powerBuffer)
{
	int largestPower,i;
	for ( i = 0;i < 10;i++)
	{
		if (powerBuffer[i] == -1)
		{
			return largestPower;
		}
		else if (powerBuffer[i] != 0)
		{
			largestPower = powerBuffer[i];
		}
	}
	return largestPower;
}
void numToPowerBuffer(int power,int *powerBuffer)
{
	int i = 0,n;
	n = power;
	while( n > 0) 
	{ 
		if (n % 2 == 1)
		{
			powerBuffer[i] =(int)  pow(2,i);
		}
		else
		{
			powerBuffer[i] = 0;
		}
		n = n/2 ; 
		i++; 
	}
}


int arrayContains(int *buffer,int value)
{
	int i;
	for (i = 0; i < 16; i++)
	{
		if (buffer[i] == -1)
		{
			break;
		}
		else if ( buffer[i] == value)
		{
			return i;
		}
	}
	return -1;
}

void copyMatrix(struct sMatrix *matrixOut,struct sMatrix *matrixIn)
{
	memcpy(&(matrixOut->matrix[0][0]),&(matrixIn->matrix[0][0]),MATRIX_ROWS * MATRIX_COLS);
	matrixOut->rows = matrixIn->rows;
	matrixOut->cols = matrixIn->cols;
}

void aggregateMatrixProducts(struct sMatrix *matrixOut,struct sMatrix *workMatrix,int numProcessors,int rank)
{
	int i;
	MPI_Status status;
	if (rank == 0)
	{
		//Get partial results from all nodes.
		//TODO - modify to use MPI_Reduce.
		//MPI_Reduce(matrix1,matrixOut,1,matrixAdd,

		for (i =1; i< numProcessors;i++)
		{
			clearMatrix(workMatrix);
			printf("\nReading matrix from node %d..",i);
			MPI_Recv(&(workMatrix->matrix[0][0]),MATRIX_ROWS*MATRIX_COLS,MPI_DOUBLE,i,i,MPI_COMM_WORLD,&status);
			addMatrix(workMatrix,matrixOut);
			printf("done\n");
		}
	}
	else
	{
		printf("\nNode %d:Now returning matrix..",rank);
		//fflush(stdout);
		MPI_Send(&(matrixOut->matrix[0][0]),MATRIX_ROWS*MATRIX_COLS,MPI_DOUBLE,0,rank,MPI_COMM_WORLD);
		printf("done\n");
	}
}

void addMatrix(struct sMatrix *mat1,struct sMatrix *matOut)
{
	int i,j;
	//TODO: Check dimensions of mat1 and matout
	#pragma omp parallel for
	for(i = 0; i < mat1->rows; i++)
	{
		for (j = 0; j < mat1->cols;j++)
		{
			matOut->matrix[i][j] += mat1->matrix[i][j];
		}
	}
}
void clearMatrix(struct sMatrix *matrixOut)
{
	/*int i,j;
	#pragma omp parallel for
	for(i = 0; i < matrixOut->rows;i++)
	{
		for (j = 0 ; j< matrixOut->cols;j++)
		{
			matrixOut->matrix[i][j] = 0;
		}
	}*/
	memset(&(matrixOut->matrix[0][0]),0,MATRIX_ROWS * MATRIX_COLS);
}
void multiplyMatrix(struct sMatrix *mat1,struct sMatrix *mat2,struct sMatrix *mat3,int numProcessors,int rank)
{
	int i,j,k;
	int start_row,end_row;

	double time1,time2;
	mat3->rows = mat1->rows;
	mat3->cols = mat2->cols;

	//allocMatrixStruct(mat3,mat1->rows,mat2->cols);

	time1 = MPI_Wtime();

	start_row = 0 + rank * (mat1->rows/numProcessors);
	end_row = start_row + (mat1->rows/numProcessors);

	for (i = start_row; i < end_row; i++)
	{
		//k look has been intentionally moved out for better locality of reference.
		for ( k = 0; k < mat1->cols;k++)
		{
			//#pragma omp parallel for
			for (j = 0; j < mat2->cols; j++)
			{
				mat3->matrix[i][j] += (mat1->matrix[i][k] * mat2->matrix[k][j]);
			}
		}

		/*if ( i % 50 == 0)
		{
			time2 = time(NULL);
			//printf("\r %.2f percent done (%f sec/row)..", (float)((float) i * 100 / mat1->rows),((float) 50/(time2-time1)));
			time1 = time2;
		}*/
	}
	time2 = MPI_Wtime();
	printf("\nRank %d did %d rows in %4.2lf seconds.\n",rank,(end_row - start_row),(time2 - time1));
}




int readCommandlineArgs(int argc,char **argv,int* power,char *folderName)
{
	int i;
	for (i = 1; i < argc;i++)
	{
		if (strcmp(argv[i],"-pwr") == 0)
		{
			if ( i < argc)
			{
				i++;
				*power = atoi(argv[i]);
				if (*power == 0)
				{
					return -1;
				}
			}
			else
			{
				return -1;
			}
		}
		else if ( strcmp(argv[i],"-d") == 0)
		{
			if ( i < argc)
			{
				i++;
				if (strlen(argv[i]) > 0)
				{
					strcpy(folderName,argv[i]);
				}
				else
				{
					return -1;
				}
			}
			else
			{
				return -1;
			}
		}
	}
	return 0;
}

int main(int argc,char **argv)
{
	int i,retCode,rank,numProcessors,power;
	struct sMatrix matrix1,matrixOut,tempMatrix;
	char folderPath[128] = {'\0'};
	char inFileName[128] = {'\0'};
	char outFilename[128] = {'\0'};
	int currPower = 1,largestPower;
	int powerBuffer[16] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
	struct sMatrix *partialResultBuffer;
	int partialBufferLen = 0,partialBufferPtr = 0;
	double time1,time2;

	i  = 0;

	MPI_File fin,fout;
	MPI_Status status;
	
	
	retCode = MPI_Init(&argc,&argv);

	if (retCode != MPI_SUCCESS)
	{
		printf("\nError during init! RetCode - %d\n",retCode);
		MPI_Abort(MPI_COMM_WORLD,retCode);
	}
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcessors);

	if (rank == 0)
	{
		time1 = MPI_Wtime();
	}

	allocMatrixStruct(&matrix1,MATRIX_ROWS,MATRIX_COLS);
	allocMatrixStruct(&matrixOut,MATRIX_ROWS,MATRIX_COLS);
	allocMatrixStruct(&tempMatrix,MATRIX_ROWS,MATRIX_COLS);

	retCode = readCommandlineArgs(argc,argv,&power,folderPath);
	if (rank == 0)
	{
		printf("\nInitialized %d processors.\n",numProcessors);

		if ( argc > 1)
		{
			if (retCode < 0)
			{
				printf("\nIncorrect usage.\n");
				MPI_Abort(MPI_COMM_WORLD,MPI_ERR_OTHER);
				return 0;
			}
		}
		else 
		{
			printf("\nMissing command-line arguments.\n");
			MPI_Abort(MPI_COMM_WORLD,MPI_ERR_OTHER);
			return 0;
		}
		if (power < 2)
		{
			if (rank == 0)
			{
				printf("\nPower should be atleast 2.\n");
				MPI_Abort(MPI_COMM_WORLD,MPI_ERR_OTHER);
				return 0;
			}
		}
		if (folderPath[strlen(folderPath)-1] != '/')
		{
			//Folder path doesnt have trailing slash. Add it
			sprintf(inFileName,"%s/%s",folderPath,INPUT_FILE);
		}
		else
		{
			sprintf(inFileName,"%s%s",folderPath,INPUT_FILE);
		}
	
		MPI_File_open(MPI_COMM_SELF,inFileName,MPI_MODE_RDONLY,MPI_INFO_NULL,&fin);

		if (fin != NULL)
		{
			MPI_File_read(fin,&(matrix1.matrix[0][0]),MATRIX_ROWS * MATRIX_COLS,MPI_DOUBLE,&status);
			MPI_File_close(&fin);
		}
		else
		{
			printf("\n Cant open %s for reading.\n",inFileName);
			MPI_Abort(MPI_COMM_WORLD,MPI_ERR_OTHER);
		}
	}
	numToPowerBuffer(power,powerBuffer);
	partialBufferLen = getNumPartialResults(powerBuffer);
	if (partialBufferLen > 1)
	{
		partialResultBuffer = (struct sMatrix *) malloc (sizeof(struct sMatrix) * partialBufferLen);
		for (i = 0; i < partialBufferLen; i++)
		{
			allocMatrixStruct(&partialResultBuffer[i],MATRIX_ROWS,MATRIX_COLS);
		}
	}

	largestPower = getLargestPartialResult(powerBuffer);
	if (rank == 0)
	{
		printf("\nLargest Power - %d",largestPower);
		printf("\nPower buffer - ");
		for (i =0 ;i<16;i++)
		{
			if (powerBuffer[i] == -1)
			{
				break;
			}
			else
			{
				printf(" %d ",powerBuffer[i]);
			}
		}
	}

	//Special case for A^1
	if (rank == 0 )
	{
		if (partialBufferLen > 1)
		{
			if (arrayContains(powerBuffer,1) > -1)
			{
				printf("\nSaving result(%d) in partialResultBuffer.",1);
				copyMatrix(&(partialResultBuffer[partialBufferPtr]),&matrix1);
				partialBufferPtr++;
			}
		}
	}
	while (currPower < largestPower )
	{
		clearMatrix(&matrixOut);
		multiplyMatrix(&matrix1,&matrix1,&matrixOut,numProcessors,rank);
		currPower = currPower * 2;
		printf("\nNode %d: Done with multiplication. CurrPower - %d. Waiting at the barrier.",rank,currPower);
		MPI_Barrier(MPI_COMM_WORLD);
		
		aggregateMatrixProducts(&matrixOut,&tempMatrix,numProcessors,rank);
		if (rank == 0 )
		{
			if (partialBufferLen > 1)
			{
				if (arrayContains(powerBuffer,currPower) > -1)
				{
					printf("\nSaving result(%d) in partialResultBuffer.",currPower);
					copyMatrix(&(partialResultBuffer[partialBufferPtr]),&matrixOut);
					partialBufferPtr++;
				}
			}
		}
		copyMatrix(&matrix1,&matrixOut);
	}

	//Multiply all the partial results
	if (partialBufferLen > 1)
	{
		for (i = 0; i < partialBufferLen; i++)
		{
			if (matrixOut.matrix[0][0] != 0.00)
			{
				copyMatrix(&matrix1,&matrixOut);
				clearMatrix(&matrixOut);
			}
			if (rank == 0)
			{
				printf("\nCalculating partial result %d",i);
			}
			MPI_Bcast(&(partialResultBuffer[i].matrix[0][0]),MATRIX_ROWS * MATRIX_COLS,MPI_DOUBLE,0,MPI_COMM_WORLD);
			MPI_Bcast(&(partialResultBuffer[i].rows),1,MPI_INT,0,MPI_COMM_WORLD);
			MPI_Bcast(&(partialResultBuffer[i].cols),1,MPI_INT,0,MPI_COMM_WORLD);
			multiplyMatrix(&matrix1,&(partialResultBuffer[i]),&matrixOut,numProcessors,rank);
			aggregateMatrixProducts(&matrixOut,&tempMatrix,numProcessors,rank);
		}
	}

	if (rank == 0)
	{

		if (folderPath[strlen(folderPath)-1] != '/')
		{
			//Folder path doesnt have trailing slash. Add it
			sprintf(outFilename,"%s/C-%d.dat",folderPath,power);
		}
		else
		{
			sprintf(outFilename,"%sC-%d.dat",folderPath,power);
		}
		MPI_File_open(MPI_COMM_SELF,outFilename,MPI_MODE_WRONLY | MPI_MODE_CREATE,MPI_INFO_NULL,&fout);
		if (fout != NULL)
		{
			MPI_File_write(fout,&(matrixOut.matrix[0][0]),MATRIX_ROWS * MATRIX_COLS,MPI_DOUBLE,&status);
			MPI_File_close(&fout);
		}
		else
		{
			printf("\n Cant open %s for writing.\n",outFilename);
			MPI_Abort(MPI_COMM_WORLD,MPI_ERR_OTHER);
		}
	}

	freeMatrixStruct(&matrix1);
	freeMatrixStruct(&matrixOut);
	freeMatrixStruct(&tempMatrix);
	
	if (rank == 0)
	{
		if (partialBufferLen > 1)
		{
			for (i = 0; i < partialBufferLen; i++)
			{
				freeMatrixStruct(&partialResultBuffer[i]);
			}
		}
	}
	if (rank == 0)
	{
		time2 = MPI_Wtime();
	}
	MPI_Finalize();
	if (rank == 0)
	{
		printf("\nTotal Execution Time -  %4.2lf seconds.\n",(time2-time1));
	}
	return 0;
}
