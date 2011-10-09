#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

//#define __DEBUG

#define element_addr(a, m, n, d) (a + ((m) * (d) + n))
#define element(a, m, n, d) (((m >= 0)&&(m < d)&&(n >= 0)&&(n < d))? (a[(m) * (d) + n]) : 0) 

#define CUDA_CALL(cmd) do { \
	if((err = cmd) != cudaSuccess) { \
		printf("(%d) Cuda Error:(%d) %s\n", __LINE__,int(err), cudaGetErrorString(err) ); \
	} \
} while(0)


#define BLK_SZ 16
#define BLK_SIDE 4

__global__ void computeKernel(int *living, float *honeys[2], float *honeyr, int d, float rbee, float rflow) {
	//honeyr[threadIdx.x] = honeys[0][threadIdx.x];
	//honeyr[threadIdx.x] = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//honeyr[i*d + j] = i+j;
	*(element_addr(honeyr, i, j, d)) = element(honeyr,i-1,j-1,d);
}


__global__ void computeKernelReal(int *living, float *honeyin,float *honeyout, int d, float rbee, float rflow) {
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	*(element_addr(honeyin, i, j, d)) = rflow * (element(honeyin, i-1, j-1, d) + element(honeyin, i-1, j, d) + element(honeyin, i-1, j+1, d) + element(honeyin, i, j-1, d)   + element(honeyin, i, j+1, d) + element(honeyin, i+1, j-1, d) + element(honeyin, i+1, j, d) + element(honeyin, i+1, j+1, d) ) + (1.0 - 8.0 * rflow) * element(honeyin, i, j, d) + rbee * element(living, i, j, d);
} 

int calculateGPU(const int *living, float *honey[2], int d, int n, float rbee, float rflow)
{
	cudaError_t err;
	clock_t start, end;
	cudaEvent_t kstart, kstop;
	float ktime;
	double time;

	int i;


	/* PA2: Define your local variables here */
	int *living_d;
	float *honeyin_d;
	float *honey_r;

	/* Set up device timers */
	#line 49
	CUDA_CALL(cudaSetDevice(0));
	#line 50
	CUDA_CALL(cudaEventCreate(&kstart));
	#line 53
	CUDA_CALL(cudaEventCreate(&kstop));

	/* Start GPU end-to-end timer */
	start = clock();

	/* PA2: Add CUDA kernel call preparation code here */
	#line 60
	CUDA_CALL(cudaMalloc((void **)&living_d, d * d * sizeof(int)));
	#line 62
	CUDA_CALL(cudaMalloc((void **)&honeyin_d, d * d * sizeof(float)));
	#line 66
	CUDA_CALL(cudaMalloc((void **)&honey_r, d * d * sizeof(float)));
	#line 68
	CUDA_CALL(cudaMemcpy(living_d, living, d * d * sizeof(int), cudaMemcpyHostToDevice));
	#line 70
	CUDA_CALL(cudaMemcpy(honeyin_d, honey[0], d * d * sizeof(float), cudaMemcpyHostToDevice));

	/* Start GPU computation timer */
	#line 76
	CUDA_CALL(cudaEventRecord(kstart, 0));

	/* PA2: Add main honey level simulation loop here */
	dim3 dimGrid(d/BLK_SIDE,d/BLK_SIDE);
	dim3 dimBlock(BLK_SIDE,BLK_SIDE);
	for (i=0;i< n;i++)
	{
		//printf("Calling kernel with resin-%d and src-%d\n",resin,src);
		computeKernelReal<<<dimGrid,dimBlock>>>(living_d,honeyin_d,honey_r,d,rbee,rflow);
		#line 4848
		//CUDA_CALL(cudaThreadSynchronize());
		CUDA_CALL(cudaMemcpy(honeyin_d,honey_r,d * d * sizeof(float),cudaMemcpyDeviceToDevice ));
	}

	//computeKernel<<<dimGrid,dimBlock>>>(living_d,honey_d,honey_r,d,rbee,rflow);


	/* Stop GPU computation timer */
	#line 88
	CUDA_CALL(cudaEventRecord(kstop, 0));
	#line 90
	CUDA_CALL(cudaEventSynchronize(kstop));

	#line 93
	CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
	printf("GPU computation: %f msec\n", ktime);



	/* PA2: Add post CUDA kernel call processing and cleanup here */
	#line 98
	//CUDA_CALL(cudaMemcpy(honey[0],honey_d[resin],d * d * sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(honey[1],honey_r,d * d * sizeof(float),cudaMemcpyDeviceToHost));

	printf("\nhoney[] after cuda kernel call -\n");
	for(int i = 0; i < d; i++ ) {
		for(int j = 0; j < d; j++ ) {
			printf("%f ", element(honey[1], i, j, d));
		}
		printf("\n");
	}
	
	#line 100
	CUDA_CALL(cudaFree(living_d));
	#line 102
	CUDA_CALL(cudaFree(honeyin_d));
	#line 106
	CUDA_CALL(cudaFree(honey_r));

	/* Stop GPU end-to-end timer and timer cleanup */
	end = clock();
	#line 111
	CUDA_CALL(cudaEventDestroy(kstart));
	#line 113
	CUDA_CALL(cudaEventDestroy(kstop));
	time = ((double)(end-start))/CLOCKS_PER_SEC;
	printf("GPU end-to-end: %lf sec\n", time);
	return 1;
}
