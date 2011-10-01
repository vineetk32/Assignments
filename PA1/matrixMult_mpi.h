#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <mpio.h>
#include <math.h>
#include <omp.h>

//#define INPUT_FILE "/gpfs_share/csc548/vineet/A.dat"
#define INPUT_FILE "A.dat"
#define OUTPUT_FILE "C.dat"

#define MATRIX_ROWS 4096
#define MATRIX_COLS 4096


struct sMatrix
{
	double **matrix;
	int rows;
	int cols;
};

void allocMatrixStruct(struct sMatrix *matrix,int rows,int cols);
void multiplyMatrix(struct sMatrix *mat1,struct sMatrix *mat2,struct sMatrix *mat3,int numProcessors,int rank);
void addMatrix(struct sMatrix *mat1,struct sMatrix *matOut);
int  readCommandlineArgs(int argc,char **argv,int* power,char *folderName);
void numToPowerBuffer(int power,int *powerBuffer);
void clearMatrix(struct sMatrix *matrixOut);
int arrayContains(int *buffer,int value);
void copyMatrix(struct sMatrix *matrixOut,struct sMatrix *matrixIn);
int getNumPartialResults(int *powerBuffer);
int getLargestPartialResult(int *powerBuffer);
void aggregateMatrixProducts(struct sMatrix *matrixOut,struct sMatrix *workMatrix,int numProcessors,int rank);
void deallocMatrixStruct(struct sMatrix *matrix);
