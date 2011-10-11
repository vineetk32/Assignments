// This program performs a Gaussian elimination method for solving a set of linear
// equations of the following form AX = Y
// The code is greatly simplified to help students focusing on key loop structures.
// Some corner cases are not handled, for example, pivoting when the pivot element
// A[pivot][pivot] is zero generally involves swapping two rows of the matrix. 


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include "defines.h"

#define MATRIX_DIM 2000
#define SEED 0

double A[MATRIX_DIM][MATRIX_DIM];
double X[MATRIX_DIM];
double Y[MATRIX_DIM];

void printMatrix() 
{
	int i, j;
	for (i=0; i<MATRIX_DIM; i++) {
		for (j=0; j<MATRIX_DIM; j++) {
			printf("%7.3f ", A[i][j]);
		}
		printf(" | %7.3f \n", Y[i]);
	}
}

void printSol() 
{
	int i, j;
	printf("--- Solution of Y=AX: \n");
	for (i=0; i<MATRIX_DIM; i++) {
		printf(" X%d = %7.3f \n", i, X[i]);
	}
}

initMatrix()
{
	int i, j;

	srand(SEED);
	for (i=0; i<MATRIX_DIM; i++) {
		for (j=0; j<MATRIX_DIM; j++) {
			A[i][j] =  100 * ((double)rand() / ((double)(RAND_MAX)+(double)(1)) ) ;      
		}
		Y[i] = 100 * ((double)rand() / ((double)(RAND_MAX)+(double)(1)) ) ;      
	}
}

int main() 
{
	double adjust;
	int i, j, pivot;
	struct timespec start_time,end_time;

	// initialize matrix
	initMatrix();

	clock_gettime(CLOCK_MONOTONIC,&start_time);
	// upper triangulation step
	for (pivot=0; pivot<MATRIX_DIM-1; pivot++) {
		for (i=pivot+1; i<MATRIX_DIM; i++) {
			adjust = A[i][pivot] / A[pivot][pivot];
#pragma omp parallel for private(j) shared(adjust,i,A)  schedule(SCHEDULE_TYPE,SCHEDULE_CHUNKSIZE)
			for (j=pivot; j<MATRIX_DIM; j++) {
				A[i][j] = A[i][j] - adjust * A[pivot][j];
			}
			Y[i] = Y[i] - adjust * Y[pivot];
		}
	}

	clock_gettime(CLOCK_MONOTONIC,&end_time);


	// diagonalization step
	for (pivot=MATRIX_DIM-1; pivot>0; pivot--) {
		for (i=pivot-1; i>=0; i--) {
			adjust = A[i][pivot] / A[pivot][pivot];
			A[i][pivot] = A[i][pivot] - adjust * A[pivot][pivot];   
			Y[i] = Y[i] - adjust * Y[pivot];
		}
	}

	// solution step
	for (i=0; i<MATRIX_DIM; i++)
		X[i] = Y[i] / A[i][i];

	//printSol();
	printf("\nTotal execution time for parallel gauss_j - %lf s\n",(end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0);
}
