#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CALL(cmd) do { \
	if((err = cmd) != cudaSuccess) { \
		printf("(%d) Cuda Error: %s\n", __LINE__, cudaGetErrorString(err) ); \
	} \
} while(0)


// Kernel definition 
__global__ void VecAdd(float* A, float* B, float* C) 
{ 
    int i = threadIdx.x; 
    C[i] = A[i] + B[i]; 
} 
 
int main() 
{ 
	//Test vars
	cudaError_t err;
	float h_a[4] = {1,2,3,4};
	float h_b[4] = {1,2,3,4};
	float h_c[4] = {0,0,0,0};
	
	float *d_a,*d_b,*d_c;
	// Kernel invocation with N threads 

	CUDA_CALL(cudaMalloc((void**)&d_a, 4 * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_b, 4 * sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&d_c, 4 * sizeof(float)));

	CUDA_CALL(cudaMemcpy(d_a, h_a, 4 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_b, h_b, 4* sizeof(float), cudaMemcpyHostToDevice));

	VecAdd<<<1,4>>>(d_a, d_b, d_c);
	cudaMemcpy(h_c, d_c, 4* sizeof(float), cudaMemcpyDeviceToHost);
	printf("\n Vector sum is %f,%f,%f,%f. ",h_c[0],h_c[1],h_c[2],h_c[3]);
	return 0;
}
