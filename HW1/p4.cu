#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_PROCESSES 12
#define THREADS_PER_BLOCK 8
#define BLOCKS_PER_GRID 32


//Modified from Dr. Xiaosong Ma's sample CUDA code provided in CSC548
#define CUDA_CALL(cmd) do { \
	if((err = cmd) != cudaSuccess) { \
		printf("(%d) Cuda Error: %s\n", __LINE__, cudaGetErrorString(err) ); \
		err = cudaSuccess; \
	} \
} while(0)

//Taken from CUDA Ref manual appendix B
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
	#define cudaprintf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__global__ void threadFunc(char *data,int offset,int *deviceCount,int *deviceLines,char *currProcess) 
{
	int _totalLines = 0,_totalWords = 0,threadID;
	long start,end,curr;
	int exitFlag = 0,i;

	threadID = blockDim.x * blockIdx.x + threadIdx.x;

	//TODO: Logic for start and offset
	start = (threadID) * offset;
	end = start + offset - 1;

	//TODO: Adjust start and end to a line boundary
	if (start != 0)
	{
		while (data[start] != '\n' && data[start] != '\0')
		{
			start++;
		}
	}
	if (data[start] == '\0')
	{
		exitFlag = 1;
	}
	//cudaprintf("\nTrying start - %d and end - %d",start,end);
	while (data[end] != '\n' && data[end] != '\0')
	{
		end++;
	}

	curr = start;
	//Read the input data between the offsets char-by-char
	while (exitFlag == 0)
	{
		while (data[curr] != '\n')
		{
			if (data[curr] == currProcess[0])
			{
				i = 0;
				while (data[curr] == currProcess[i])
				{
					curr++;
					i++;
				}
				if ( data[curr] == '[')
				{
					_totalWords++;
				}
			}
			if (data[curr] == '\0')
			{
				exitFlag = 1;
				break;
			}
			else if (curr >= end)
			{
				exitFlag = 1;
				break;
			}
			curr++;
		}
		curr++;
		_totalLines++;
	}
	deviceCount[threadID] = _totalWords;
	deviceLines[threadID] = _totalLines;
} 
int main(int argc, char **argv)
{
	FILE *flog,*fproc;
	int i,j,k,threadOffset,numProcesses,totalLines = 0,printedOnce = 0;
	char *fileContents;
	long fileSize;
	cudaError_t err;
	char tempBuff[128] = {'\0'};
	char **processList;
	int *processCount;
	float ktime;
	
	cudaEvent_t kstart, kstop;
	
	int *hostCount,*hostLines;

	char *deviceCopy,*deviceProcess;
	int *deviceCount,*deviceLines;

	int threadsPerBlock,blocksPerGrid;

#ifndef _WIN32
	struct timespec start_time,end_time;
#endif

	if (argc != 3)
	{
		printf("Incorrect usage!\nUsage - %s path-to-log-file path-to-process-file\n",argv[0]);
		return -1;
	}
	
	fproc = fopen(argv[2],"r");
	if (fproc == NULL)
	{
		printf("\nError opening processfile!\n");
		return -1;
	}
	
	i = 0;
	processList = (char **) malloc (sizeof(char *) * MAX_PROCESSES);
	while (fscanf(fproc,"%s",tempBuff) >= 0)
	{
		processList[i] = strdup(tempBuff);
		i++;
	}
	fclose(fproc);
	numProcesses = i;

	flog = fopen(argv[1],"r");
	if (flog == NULL)
	{
		printf("\nError opening logfile!\n");
		return -1;
	}
	
	if (fseek(flog,0,SEEK_END) == 0)
	{
		fileSize = ftell(flog);
	}

	fileContents = (char *) malloc (sizeof(char) * fileSize);
	rewind(flog);
	fread(fileContents,1,fileSize,flog);
	fclose(flog);

	processCount = (int *) malloc (sizeof(int) * numProcesses);
	for (j = 0; j < numProcesses; j++)
	{
		processCount[j] = 0;
	}

	//Setting up the CUDA timers.
	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaEventCreate(&kstart));
	CUDA_CALL(cudaEventCreate(&kstop));

	for (k = 16; k < 512; k *= 2)
	{
		threadsPerBlock = k;
		blocksPerGrid = BLOCKS_PER_GRID;
		
		//ThreadOffset is also the blockSize per thread.
		threadOffset = fileSize / ( threadsPerBlock * blocksPerGrid) ;

		hostCount = (int *) malloc (sizeof(int) * threadsPerBlock * blocksPerGrid);
		hostLines = (int *) malloc (sizeof(int) * threadsPerBlock * blocksPerGrid);
		
		CUDA_CALL(cudaEventRecord(kstart, 0));
		
		CUDA_CALL(cudaMalloc((void **)&deviceCopy, fileSize * sizeof(char)));
		CUDA_CALL(cudaMalloc((void **)&deviceCount,sizeof(int) * threadsPerBlock * blocksPerGrid));
		CUDA_CALL(cudaMalloc((void **)&deviceLines,sizeof(int) * threadsPerBlock * blocksPerGrid));
		CUDA_CALL(cudaMalloc((void **)&deviceProcess,sizeof(char) * 32));

		CUDA_CALL(cudaMemcpy(deviceCopy,fileContents, fileSize * sizeof(char), cudaMemcpyHostToDevice));

		//Start the CPU Timer 
#ifndef _WIN32
		clock_gettime(CLOCK_MONOTONIC,&start_time);
#endif
		totalLines = 0;
		//make a kernel for each process in the processList.
		for (i = 0; i < numProcesses; i++)
		{
			for (j = 0; j < threadsPerBlock * blocksPerGrid ;j++)
			{
				hostCount[j] = 0;
				hostLines[j] = 0;
			}
			
			CUDA_CALL(cudaMemcpy(deviceProcess,processList[i], sizeof(char) * 32,cudaMemcpyHostToDevice));
			CUDA_CALL(cudaMemcpy(deviceCount,hostCount, sizeof(int) * threadsPerBlock * blocksPerGrid,cudaMemcpyHostToDevice));
			CUDA_CALL(cudaMemcpy(deviceLines,hostLines, sizeof(int) * threadsPerBlock * blocksPerGrid, cudaMemcpyHostToDevice));

			threadFunc<<<blocksPerGrid,threadsPerBlock>>>(deviceCopy,threadOffset,deviceCount,deviceLines,deviceProcess);
			
			CUDA_CALL(cudaMemcpy(hostCount,deviceCount, sizeof(int) * threadsPerBlock * blocksPerGrid,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(hostLines,deviceLines, sizeof(int) * threadsPerBlock * blocksPerGrid, cudaMemcpyDeviceToHost));

			//Record the time
			CUDA_CALL(cudaEventRecord(kstop, 0));
			CUDA_CALL(cudaEventSynchronize(kstop));
			CUDA_CALL(cudaEventElapsedTime(&ktime, kstart, kstop));
		
			//printf("GPU computation: %f msec\n", ktime);
			
			for ( j = 0; j < threadsPerBlock * blocksPerGrid; j++)
			{
				processCount[i] += hostCount[j];
				totalLines += hostCount[j];
			}
		}
#ifndef _WIN32
		clock_gettime(CLOCK_MONOTONIC,&end_time);
#endif

		//blockSize: numThreads: totalCount: CPUTime: GPUTime
		if (printedOnce == 0)
		{
			for(i = 0; i < numProcesses;i++)
			{
				printf("\npName:%s count:%d",processList[i],processCount[i]);
			}
			printf("\nTotal Number of loglines: %d\n",totalLines);
			printedOnce = 1;
		}
#ifndef _WIN32
		printf("\n%d: %d: %d: %lfms: %fms\n",threadOffset,threadsPerBlock * blocksPerGrid,totalLines,(end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_nsec - start_time.tv_nsec)/1000000.0,ktime);
#else
		printf("\n%d: %d: %d: %fms\n",threadOffset,threadsPerBlock * blocksPerGrid,totalLines,ktime);
#endif
	}
	return 0;
}
