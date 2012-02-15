/*
CSC501 - Operating System - Spring 2012 - North Carolina State University

HomeWork2 Prob4. See - http://courses.ncsu.edu/csc501/lec/001/hw/hw2/
Author: Salil Kanitkar (sskanitk@ncsu.edu)

For Compiling - 
$ make clean ; make a4
For Executing -
$ ./a4 <path-to-log-file> <path-to-process-list-file>
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <cuda_runtime.h>
#include<sys/types.h>
#include <math.h>

#ifndef _WIN32
#include<sys/time.h>
#endif

/* Uncomment the below line to enable debug prints 
*/
//#define VERBOSE 1

#define MAX_LOGFILE_SIZE (1<<20)
#define MAX_LOGLINE_SIZE 100
#define MAX_PROC_NUM 10
#define MAX_PNAME_LEN 50
#define MAX_NUM_THREADS 100*500
#define MAX_NUM_BLOCKS 100
#define MAX_THREADS_PER_BLOCK 400

/* struct to hold process names read from the proclistfile. */
typedef struct _proc_entry_t {
	char pname[MAX_PNAME_LEN];
	int count;
}proc_entry_t;

/* struct for each thread to put the data calculated by it. */
typedef struct _stats_entry_t {
	proc_entry_t proclist[MAX_PROC_NUM];
}stats_entry_t;

/* CUDA device local func for string copy. */
__device__ void dev_mystrcpy(char *t, char *s)
{
	while ( *s != '\0' ) {
		*t++ = *s++;
	}
	*t = '\0';
}

/* CUDA device local func for getting string length. */
__device__ int dev_my_strlen(char *src)
{
	int len=0;
	while ( *src++ != '\0' )
		len++;
	return (len);
}

/* CUDA device func for comparing strings. */
__device__ int dev_my_strcmp(char *s, char *d)
{
	int len = dev_my_strlen(s), tmplen = dev_my_strlen(d);
	int i=0;

	if (len != tmplen)
		return 1;

	while (i < len) {
		if (*(s+i) != *(d+i))
			return 1;
		i += 1;
	}

	return 0;	
}

/* The global kernel func. 
For the block that a thread is supposed to work with, the below function will calculate the results and populate the corresponding cell
in the dev_stats memory array.
*/
__global__ void dev_calc_stats(char *dev_fileBuf, int *dev_blockStart, int *dev_blockEnd, int numProcs, stats_entry_t *dev_stats, int paddedFileSize, int fileSize)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i=0, j=0, k=0, bufSize;
	char buf[10000], logline[MAX_LOGLINE_SIZE], tmp[MAX_PNAME_LEN];
	dev_mystrcpy(buf, "");
	dev_mystrcpy(logline, "");
	dev_mystrcpy(tmp, "");

	if (dev_blockStart[idx] > fileSize || dev_blockEnd[idx] > paddedFileSize || dev_blockStart[idx] >= dev_blockEnd[idx])
		return;

	for (i=dev_blockStart[idx] ; i <= paddedFileSize && i <= dev_blockEnd[idx] ; i++) {
		buf[j++] = dev_fileBuf[i];
	}
	buf[j] = '\0';
	bufSize = j;
	
	i = 0; j = 0;
	for (i=0 ; i < bufSize ; i++) {
		if (buf[i] == '\n') {
			if (j <= MAX_LOGLINE_SIZE)
				logline[j] = '\0';
			else {
				j = 0;
				continue;
			}
			k = 0;
			while (k+16 < 100 && logline[k+16] != '[') {
				tmp[k] = logline[k+16];
				k += 1;
			}
			tmp[k] = '\0';
			for (j=0 ; j < numProcs ; j++) {
				if (dev_my_strcmp(dev_stats[idx].proclist[j].pname, tmp) == 0) 
					dev_stats[idx].proclist[j].count += 1;
			}
			j = 0;
		}
		else {
			if (j < MAX_LOGLINE_SIZE)
				logline[j] = buf[i];
			j += 1;
		}
	}

}

__global__ void reducerFunc(stats_entry_t *input_stats,stats_entry_t *output_stats,int numProcesses,int totalThreads)
{
	int j = 0;
	unsigned int myID = blockIdx.x*blockDim.x + threadIdx.x;
	//extern __shared__ stats_entry_t shared_stats[][];

	/*for (j = 0; i < numProcesses; j++)
	{
		shared_stats[tid].proclist[j].count = input_stats[i].proclist[j].count;
	}
	
	__syncthreads();*/

	// do reduction in shared mem
	if (totalThreads > 1 && (myID *2 + 1) < totalThreads)
	{
		for (j = 0; j < numProcesses; j++)
		{
			output_stats[myID].proclist[j].count = input_stats[2*myID].proclist[j].count + input_stats[2*myID + 1].proclist[j].count;
		}
	}
}


int main(int argc, char *argv[])
{
	FILE *fp_logfile, *fp_proclist;
	char *fileBuf=(char *)malloc(sizeof(char)*MAX_LOGFILE_SIZE);
	char *procBuf=(char *)malloc(sizeof(char)*MAX_PROC_NUM*MAX_PNAME_LEN);
	int numThreads=0, numBlocks=0, numThreadsPerBlock=0,  paddedFileSize=0, blockSize=0;
	long fileSize = 0,i = 0;
	int *blockStart=0, *blockEnd=0;
	int numProcs, count, tot_count, pflag=0,done = 0;
	int reducerBlocks, reducerThreadsPerBlock;
	int j, k, start,blockLoop,threadLoop;
	stats_entry_t *stats=0;
	char *pname, proclist[MAX_PROC_NUM][MAX_PNAME_LEN];

	char *dev_fileBuf;
	int *dev_blockStart, *dev_blockEnd;
	stats_entry_t *dev_stats,*dev_reducer_stats;

#ifndef _WIN32
	struct timeval t_start, t_end;
#endif
	cudaEvent_t dev_t_start, dev_t_end;
	float time_elapsed;

	if (argc != 3) {
		printf("Usage: ./log_stats path-to-log-file path-to-process-list-file\n");
		exit(1);
	}

	if (!(fp_logfile = fopen(argv[1], "r"))) {
		printf("Error opening Log File!\n");
		exit(1);
	}

	if (!(fp_proclist = fopen(argv[2], "r"))) {
		printf("Error opening Process listing file!\n");
		exit(1);
	}

	/* Read up the proclistfile in a local buffer in memory. */
	i = 0;
	for (i=0 ; !feof(fp_proclist) ; ) {
		i += fread(&(procBuf[i]), 1, 1, fp_proclist);
	}

	/* Read up the entire logfile in a local buffer in memory. */
	i = 0;
	for (i=0 ; !feof(fp_logfile) ; ) {
		i += fread(&(fileBuf[i]), 1, 1, fp_logfile);
	}
	fileSize = i;

#ifdef VERBOSE
	printf("procListFile:\n%s", procBuf);
#endif

	/* Extract out all the process names from proclistfile and populate the proclist array. */
	i = 0;
	pname = strtok(procBuf, "\n");
	while (pname) {
		strcpy(proclist[i], pname);
		i++;
		pname = strtok(NULL, "\n");
	}
	numProcs = i;

#ifdef VERBOSE
	printf("numProcs:%d\n", numProcs);
	for (i=0 ; i < numProcs ; i++) {
		printf("%s\n", proclist[i]);
	}
#endif

	if (fileSize < 65536) {
		cudaMalloc((void **)&dev_fileBuf, sizeof(char)*MAX_LOGFILE_SIZE);
		cudaMemset((void *)dev_fileBuf, 0, sizeof(char)*MAX_LOGFILE_SIZE);

		cudaMalloc((void **)&dev_blockStart, sizeof(int)*MAX_NUM_THREADS);
		cudaMemset((void *)dev_blockStart, 0, sizeof(int)*MAX_NUM_THREADS);

		cudaMalloc((void **)&dev_blockEnd, sizeof(int)*MAX_NUM_THREADS);
		cudaMemset((void *)dev_blockEnd, 0, sizeof(int)*MAX_NUM_THREADS);

		cudaMalloc((void **)&dev_stats, sizeof(stats_entry_t)*MAX_NUM_THREADS);
		cudaMemset((void *)dev_stats, 0, sizeof(stats_entry_t)*MAX_NUM_THREADS);

		cudaMalloc((void **)&dev_reducer_stats, sizeof(stats_entry_t)*MAX_NUM_THREADS);
		cudaMemset((void *)dev_reducer_stats, 0, sizeof(stats_entry_t)*MAX_NUM_THREADS);

		blockStart = (int *)malloc(sizeof(int)*MAX_NUM_THREADS);
		blockEnd = (int *)malloc(sizeof(int)*MAX_NUM_THREADS);

		stats = (stats_entry_t *)malloc(sizeof(stats_entry_t)*(MAX_NUM_THREADS));
	}

	for (blockLoop = 1; pow((float)2,blockLoop) <  MAX_NUM_BLOCKS; blockLoop++)
	{
		numBlocks = pow((float)2,blockLoop);

		/* Vary the number of threads per block by some offset. */
		for (threadLoop = 1; pow((float)2,threadLoop) <  MAX_THREADS_PER_BLOCK; threadLoop++)
		{
			numThreadsPerBlock = pow((float)2,threadLoop);

			//numBlocks = 25 ; numThreadsPerBlock = 324;
			/* The actual number of threads to be used for this run of the program. */
			numThreads = numBlocks * numThreadsPerBlock;

			if (fileSize > 65535) {
				blockStart = (int *)malloc(sizeof(int)*MAX_NUM_THREADS);
				blockEnd = (int *)malloc(sizeof(int)*MAX_NUM_THREADS);
				stats = (stats_entry_t *)malloc(sizeof(stats_entry_t)*(MAX_NUM_THREADS));
			}

			for (i=0 ; i < MAX_NUM_THREADS ; i++) {
				blockStart[i] = 0;
			}

			for (i=0 ; i < MAX_NUM_THREADS ; i++) {
				blockEnd[i] = 0;
			}

			for (i=0 ; i < MAX_NUM_THREADS ; i++)  {
				for (j=0 ; j < numProcs ; j++) {
					strcpy(stats[i].proclist[j].pname, "");
					stats[i].proclist[j].count = 0;
				}
			}

			for (i=0 ; i < numThreads ; i++) {
				for (j=0 ; j < numProcs ; j++) {
					strcpy(stats[i].proclist[j].pname, proclist[j]);
					stats[i].proclist[j].count = 0;
				}
			}

			/* Do padding etc. Adjust the length. */
			paddedFileSize = fileSize;
			blockSize = (int)fileSize/numThreads;

			if ( fileSize%numThreads != 0 ) {
				paddedFileSize = fileSize + (numThreads - (fileSize%numThreads));
				blockSize = (int)paddedFileSize/numThreads;
				memset(&(fileBuf[fileSize]), 0, paddedFileSize - fileSize);
			}

			if (blockSize < 20 || blockSize >= 10000) { ;
			/* If the blockSize falls below 20, then no single block can contain any process name. So skip this invocation. 
			Uncomment the below line to display the corresponding message in the program output.
			*/
			/* printf("blockSize:%d numThreads:%d - No legal processing possible for this configuration.!\n", blockSize, numThreads);*/
			continue;
			}
#ifdef VERBOSE
			printf("LogFile:\n%s\n", fileBuf);

			printf("fileSize:%d paddedFileSize:%d blockSize:%d\n\n", fileSize, paddedFileSize, blockSize);
#endif

			int x; 
			//int activeThreads;
			/* Build up blockStart and blockEnd arrays. They will keep track of start and end of every block for this run. */
			for (i=0, k=0, start=0 ; i < numThreads; i++, j++) {

				blockStart[i] = start;
				k = 0;

				if (start+blockSize >= paddedFileSize) {
					blockEnd[i] = paddedFileSize;
					//activeThreads = i;
					for (x = i+1 ; x < numThreads ; x++) {
						blockStart[x] = paddedFileSize;
						blockEnd[x] = paddedFileSize;
					}
					break;
				}

				if (fileBuf[(start+blockSize)] != '\n') {
					k = 1;
					while (((start+blockSize+k) <= paddedFileSize) && (fileBuf[start+blockSize+k] != '\n'))
						k += 1;
					blockEnd[i] = start + blockSize + k;
				} else {
					blockEnd[i] = start + blockSize;
				}

				if (blockEnd[i] > paddedFileSize)
					blockEnd[i] = paddedFileSize;

				if ((blockEnd[i]+1) <= paddedFileSize)
					start = blockEnd[i] + 1;
				else
					start = paddedFileSize;
			}

#ifdef VERBOSE
			printf("Initialized Data as follows:\n");
			for (i=0 ; i < numThreads ; i++) {
				printf("Block %d\n", i);
				printf("blockStart:%d blockEnd:%d\n", blockStart[i], blockEnd[i]);
				for (j=blockStart[i] ; j<blockEnd[i] ; j++) { ;
					printf("%c", fileBuf[j]);
				}
				printf("\nStats:\n");
				for (j=0 ; j < numProcs ; j++) {
					printf("%s %d\n", stats[i].proclist[j].pname, stats[i].proclist[j].count);
				}
				printf("\n\n");
			}
#endif

			if (fileSize > 65536) {
				cudaMalloc((void **)&dev_fileBuf, sizeof(char)*MAX_LOGFILE_SIZE);
				cudaMemset((void *)dev_fileBuf, 0, sizeof(char)*MAX_LOGFILE_SIZE);

				cudaMalloc((void **)&dev_blockStart, sizeof(int)*MAX_NUM_THREADS);
				cudaMemset((void *)dev_blockStart, 0, sizeof(int)*MAX_NUM_THREADS);

				cudaMalloc((void **)&dev_blockEnd, sizeof(int)*MAX_NUM_THREADS);
				cudaMemset((void *)dev_blockEnd, 0, sizeof(int)*MAX_NUM_THREADS);

				cudaMalloc((void **)&dev_stats, sizeof(stats_entry_t)*MAX_NUM_THREADS);
				cudaMemset((void *)dev_stats, 0, sizeof(stats_entry_t)*MAX_NUM_THREADS);

				cudaMalloc((void **)&dev_reducer_stats, sizeof(stats_entry_t)*MAX_NUM_THREADS);
				cudaMemset((void *)dev_reducer_stats, 0, sizeof(stats_entry_t)*MAX_NUM_THREADS);

			}

			cudaEventCreate(&dev_t_start);
			cudaEventCreate(&dev_t_end);
			cudaThreadSynchronize();
#ifndef _WIN32
			gettimeofday(&t_start, NULL);
#endif
			/* Copy the data over to Device's Global Memory. */
			cudaMemcpy(dev_fileBuf, fileBuf, sizeof(char)*paddedFileSize, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_blockStart, blockStart, sizeof(int)*numThreads, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_blockEnd, blockEnd, sizeof(int)*numThreads, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_stats, stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyHostToDevice);

			cudaEventRecord(dev_t_start, 0);

			dev_calc_stats <<< numBlocks, numThreadsPerBlock >>> (dev_fileBuf, dev_blockStart, dev_blockEnd, numProcs, dev_stats, paddedFileSize, fileSize);

			cudaEventRecord(dev_t_end, 0);

			cudaEventSynchronize(dev_t_end);
			cudaEventElapsedTime(&time_elapsed, dev_t_start, dev_t_end );
			cudaEventDestroy(dev_t_start);
			cudaEventDestroy(dev_t_end);
			cudaThreadSynchronize();

			//cudaMemcpy(stats, dev_stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyDeviceToHost);

#ifdef VERBOSE
	        	printf("Final Data as follows:\n");
		        for (i=0 ; i < numThreads ; i++) {
        		        printf("Block %d\n", i);
                		printf("blockStart:%d blockEnd:%d\n", blockStart[i], blockEnd[i]);
		                for (j=blockStart[i] ; j<blockEnd[i] ; j++) { ;
        		                printf("%c", fileBuf[j]);
                		}
	                	printf("\nStats:\n");
	        	        for (j=0 ; j < numProcs ; j++) { ;
        	        	        printf("%s %d\n", stats[i].proclist[j].pname, stats[i].proclist[j].count);
                		}
	                	printf("\n\n");
	        	}
#endif
			done = 0;
			reducerBlocks = numBlocks;
			reducerThreadsPerBlock = numThreadsPerBlock;
			cudaMemcpy(dev_reducer_stats, dev_stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyDeviceToDevice);
			while (done == 0)
			{

				cudaMemcpy(stats, dev_stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyDeviceToHost);

				/*for (i=0 ; i < numThreads ; i++) {
					printf("\nBefore Reduction: Thread %d - ",i);
					for (j=0 ; j < numProcs ; j++) { 
						printf("%s %d\n", stats[i].proclist[j].pname, stats[i].proclist[j].count);
					}
				}*/

				if (reducerThreadsPerBlock == 1)
				{
					reducerBlocks = reducerBlocks / 2;
				}
				else
				{
					reducerThreadsPerBlock = reducerThreadsPerBlock / 2;
				}
				if (reducerThreadsPerBlock == 1 && reducerBlocks == 1)
				{
					done = 1;
				}
				//printf("Reducing %d,%d\n",reducerBlocks,reducerThreadsPerBlock);

				reducerFunc <<< reducerBlocks,reducerThreadsPerBlock >>> (dev_stats,dev_reducer_stats,numProcs,numBlocks * numThreadsPerBlock);

				cudaMemcpy(dev_stats, dev_reducer_stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyDeviceToDevice);
			}
			
			cudaMemcpy(stats, dev_stats, sizeof(stats_entry_t)*numThreads, cudaMemcpyDeviceToHost);
			/*for (i=0 ; i < numThreads ; i++) {
				printf("\nAfter Reduction Thread %d - ",i);
				for (j=0 ; j < numProcs ; j++) { 
					printf("%s %d\n", stats[i].proclist[j].pname, stats[i].proclist[j].count);
				}
			}*/
			/* Aggregate the results calculated by each block. 
			
			tot_count = 0;
			for (j=0 ; j < numProcs ; j++) {
				count = 0;
				for (i=0 ; i < numThreads ; i++) {
					count += stats[i].proclist[j].count;
				}
				if (!pflag)
					printf("pName: %s count: %d\n", stats[0].proclist[j].pname, count);
				tot_count += count;
			}

			if (!pflag)
				printf("Total Number of loglines: %d\n", tot_count);*/

			tot_count = 0;
			for (j=0 ; j < numProcs ; j++) {

				if (!pflag)
					printf("pName: %s count: %d\n", stats[0].proclist[j].pname, stats[0].proclist[j].count);
				tot_count += stats[0].proclist[j].count;
			}

			if (!pflag)
				printf("Total Number of loglines: %d\n", tot_count);

#ifndef _WIN32
			gettimeofday(&t_end, NULL);
			printf("blockSize:%d numThreads:%d totalCount:%d CPUTime:%8ld GPUTime:%f %d %d\n", blockSize, numThreads, tot_count, t_end.tv_usec - t_start.tv_usec + (t_end.tv_sec*1000000 - t_start.tv_sec*1000000),time_elapsed, numBlocks, numThreadsPerBlock);
#else
			printf("blockSize:%d numThreads:%d totalCount:%d GPUTime:%f\n", blockSize, numThreads, tot_count,time_elapsed);
#endif

			if (!pflag)
				pflag = 1;

			if (fileSize > 65536) {
				cudaFree(dev_stats);
				cudaFree(dev_blockStart);
				cudaFree(dev_blockEnd);
				cudaFree(dev_fileBuf);
			}

			if (fileSize > 65535) {
				free(blockStart);
				free(blockEnd);
				free(stats);
			}
		}
	}

	return 0;
}

