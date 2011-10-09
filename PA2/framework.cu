#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define __DEBUG

#define element_addr(a, m, n, d) (a + ((m) * (d) + n))
#define element(a, m, n, d) (((m >= 0)&&(m < d)&&(n >= 0)&&(n < d))? (a[(m) * (d) + n]) : 0) 

#define CUDA_CALL(cmd) do { \
	if((err = cmd) != cudaSuccess) { \
		printf("(%d) Cuda Error: %s\n", __LINE__, cudaGetErrorString(err) ); \
	} \
} while(0)

int calculateGPU(const int *living, float *honey[2], int d, int n, float rbee, float rflow);
int calculateCPU(const int *living, float *honey[2], int d, int n, float rbee, float rflow);

int main(int argc, char **argv)
{
	FILE *fp;
	int d, n;
	float rbee, rflow;
	int *living;
	float *honey[2], *honeyr;
	int i, j;
	int resin;

	if (argc != 2) {
		printf("sampl0 <input_file>\n");
		return 1;
	}
	
	fp = fopen(argv[1], "r");
	if(fp == NULL) {
		printf("fopen %s error\n", argv[1]);
		return 1;
	}

	fscanf(fp, "%d %d", &d, &n);
	fscanf(fp, "%f %f", &rbee, &rflow);
#ifdef __DEBUG
	printf("D=%d N=%d\n", d, n);
	printf("R(bee)=%f R(flow)=%f\n", rbee, rflow);
#endif

	living = (int *) malloc( d * d * sizeof(int) );
	if(living == NULL) {
		printf("malloc living[] error\n");
		return 1;
	}
	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			fscanf(fp, "%d", element_addr(living, i, j, d));
		}
	}
#ifdef __DEBUG
	printf("livingmap = \n");
	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			printf("%d ", element(living, i, j, d));
		}
		printf("\n");
	}
#endif

	honey[0] = (float *)malloc( d * d * sizeof(float) );
	honey[1] = (float *)malloc( d * d * sizeof(float) );
	honeyr   = (float *)malloc( d * d * sizeof(float) );
	if(honey[0] == NULL || honey[1] == NULL || honeyr == NULL ) {
		printf("malloc honey[] error\n");
		return 1;
	}

	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			fscanf(fp, "%f", element_addr(honey[0], i, j, d));
		}
	}
#ifdef __DEBUG
	printf("honey = \n");
	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			printf("%f ", element(honey[0], i, j, d));
		}
		printf("\n");
	}
#endif
	fclose(fp);

	calculateGPU(living, honey, d, n, rbee, rflow); //always return in honey[1]

	memcpy(honeyr, honey[1], d * d * sizeof(float) );

	resin = calculateCPU(living, honey, d, n, rbee, rflow);
#ifdef __DEBUG
	printf("result is in honey[%d] \n",resin);
#endif

	fp = fopen("beehive_res.txt", "w");
	n = 0;
	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			fprintf(fp, "%f ", element(honeyr, i, j, d));
			if( fabs( (element(honey[resin], i, j, d) - element(honeyr, i, j, d)) / element(honey[resin], i, j, d) ) > 0.0001) {
			//if(element(honey[resin], i, j, d) != element(honeyr, i, j, d) ) {
				printf("<%d, %d>:%f %f\n", i, j, element(honey[resin], i, j, d), element(honeyr, i, j, d));
				n++;
			}
		}
		fprintf(fp, "\n");
	}
	printf("error: %d\n", n);

	fclose (fp);

	free(honey[0]);
	free(honey[1]);
	free(honeyr);
	free(living);

	return 0;
}

int calculateCPU(const int *living, float *honey[2], int d, int n, float rbee, float rflow)
{
	int ite;
	int src, resin = 0;
	int i, j;
	clock_t start, end;
	double time;

	start = clock();
	for(ite = 0; ite < n; ite++ ) {
		src = resin;
		resin = 1 - resin;

		for(i = 0; i < d; i++ ) {
			for(j = 0; j < d; j++ ) {
				*(element_addr(honey[resin], i, j, d)) = rflow * (
						element(honey[src], i-1, j-1, d) + element(honey[src], i-1, j, d) + element(honey[src], i-1, j+1, d) 
					  +	element(honey[src], i, j-1, d)   + element(honey[src], i, j+1, d)
					  +	element(honey[src], i+1, j-1, d) + element(honey[src], i+1, j, d) + element(honey[src], i+1, j+1, d) 
					) + (1.0 - 8.0 * rflow) * element(honey[src], i, j, d)
					+ rbee * element(living, i, j, d);
			}
		}
	}
	end = clock();
	time = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("CPU computation: %lf sec\n", time);

	return resin;
}
