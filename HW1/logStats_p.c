#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "util.h"
#include <pthread.h>

#define MAX_THREADS 10
#define NUM_THREADS 6

struct threadPackage_t
{
	long startOffset;
	long endOffset;
	char **processList;
	int numProcesses;
	int *processCount;
	char *buffer;
};

int main()
{

	FILE *flog,*fproc;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	int numTokens,i = 0,j = 0,numProcesses,totalLines = 0,fileSize,threadLines;
	char *processList[MAX_PROCESSES];
	int processCount[MAX_THREADS][MAX_PROCESSES];
	long offsetArray[MAX_THREADS];
	char *fileContents;


	struct threadPackage_t threadPackage[MAX_THREADS];

	if (argc != 3)
	{
		printf("Incorrect usage!\nUsage - %s path-to-log-file path-to-process-list-file.\n",argv[0]);
		return -1;
	}

	clock_gettime(CLOCK_MONOTONIC,&start_time);

	fproc = fopen(argv[2],"r");
	if (fproc == NULL)
	{
		printf("\nError opening processfile!\n");
		return -1;
	}
	for (i = 0; i < MAX_THREADS; i++)
	{
		for (j = 0; j < MAX_PROCESSES; j++)
		{
			processCount[i][j] = 0;
		}
	}
	
	//Read every processName from the processes file
	i = 0;
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

	return 0;
}
