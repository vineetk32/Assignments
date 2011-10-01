#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>


#define INPUT_FILE "/gpfs_share/csc548/vineet/A.txt"


struct sMatrix
{
	double **matrix;
	int rows;
	int cols;
};


void allocMatrixStruct(struct sMatrix *matrix,int rows,int cols)
{
	int i;
	matrix->cols = cols;
	matrix->rows = rows;
	matrix->matrix = (double **) malloc (sizeof(double *) * rows);
	for (i = 0 ; i < rows; i++)
	{
		matrix->matrix[i] = (double *) malloc (sizeof(double) * cols);
	}
}


void multiplyMatrix(struct sMatrix *mat1,struct sMatrix *mat2,struct sMatrix *mat3)
{
	int i,j,k;
	time_t time1,time2,startTime;
	mat3->rows = mat1->rows;
	mat3->cols = mat2->cols;
	

	allocMatrixStruct(mat3,mat1->rows,mat2->cols);

	time1 = time(NULL);
	startTime = time1;
	#pragma omp parallel for
	for (i = 0; i < mat1->rows; i++)
	{
		for (j = 0; j < mat2->cols; j++)
		{
			for ( k = 0; k < mat1->cols;k++)
			{
				mat3->matrix[i][j] += (mat1->matrix[i][k] * mat2->matrix[k][j]);
			}
		}
		if ( i % 50 == 0)
		{
			time2 = time(NULL);
			printf("\r %.2f percent done (%f sec/row)..", (float)((float) i * 100 / mat1->rows),((float) 50/(time2-time1)));
			time1 = time2;
		}
	}
	printf("\nCompleted in %d seconds,\n",(time2 - startTime));
}


int main()
{
	int i,j,retCode;
	struct sMatrix matrix1,matrixOut;
	//double matrix[4096][4096];
	//char tempBuff[102400];
	char *tempBuff;
	char *tempPtr;
	FILE *fp;

	i = j = 0;
	//tempBuff[0] = '\0';

	retCode = MPI_Init(&argc,&argv);
	if (retCode != MPI_SUCCESS)
	{
		printf("\nError during init! RetCode - %d\n",retCode);
		MPI_Abort(MPI_COMM_SELF,retCode);
	}
	MPI_Comm_rank(MPI_COMM_SELF,&rank);

	MPI_File_open(MPI_COMM_SELF,argv[1],MPI_MODE_RDONLY,MPI_INFO_NULL,&handle);

	if (handle != NULL)
	{
		MPI_File_read(handle,buffer,4096 * 4096,MPI_DOUBLE,&status);
		//printf("\nFile read status is - %d",status);
		MPI_File_close(&handle);
		MPI_Finalize();

		fout = fopen(argv[2],"w");
		if (fout != NULL)
		{
			for (j = 0; j < 4096; j++)
			{
				fprintf(fout,"\n");
				for (i = 0; i< 4096; i++)
				{
					fprintf(fout,"%lf ",buffer[j * k + i]);

				}
				k++;
			}
			fclose(fout);
		}
		else
		{
			printf("\n Cant open %s for writing.\n",argv[2]);
		}

	allocMatrixStruct(&matrix1,4096,4096);

	tempBuff = (char *) malloc(sizeof(char) * 102400);
	i = 0;
	
	do
	{
		tempBuff[0] = '\0';
		fgets(tempBuff,102400,fp);
		tempPtr = strtok(tempBuff," ");
		j = 0;
		while (tempPtr != NULL)
		{
			matrix1.matrix[i][j] = atof(tempPtr);
			tempPtr = strtok(NULL," ");
			j++;
		}
		i++;
	} while (!feof(fp));
	printf("\nTotal rows read - %d\n\n",i);
	fclose(fp);
	multiplyMatrix(&matrix1,&matrix1,&matrixOut);
	
	return 0;
}
