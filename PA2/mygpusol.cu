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


#define BLK_SZ 16

__global__ void computeKernel(int *living, float *honeys, float *honeyr, int d, float rbee, float rflow) {
/* PA2: Implement your CUDA kernel here for one honey level simuation iteration */
}

int calculateGPU(const int *living, float *honey[2], int d, int n, float rbee, float rflow)
{
	cudaError_t err;
	clock_t start, end;
	cudaEvent_t kstart, kstop;
	float ktime;
	double time;

	/* PA2: Define your local variables here */

        /* Set up device timers */  
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	/* Start GPU end-to-end timer */
	start = clock();

	/* PA2: Add CUDA kernel call preperation code here */

	/* Start GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* PA2: Add main honey level simulation loop here */

	/* Stop GPU computation timer */
	CUDA_CALL(cudaEventRecord(kstop, 0));
	CUDA_CALL(cudaEventSynchronize(kstop));
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);

	/* PA2: Add post CUDA kernel call processing and cleanup here */

	/* Stop GPU end-to-end timer and timer cleanup */
	end = clock();
	CUDA_CALL(cudaEventDestroy(kstart));
	CUDA_CALL(cudaEventDestroy(kstop));
	time = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("GPU end-to-end: %lf sec\n", time);
	return 1;
}
