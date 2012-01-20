#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "util.h"
#include <pthread.h>

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256
#define BUFFER_SIZE 1024
#define MAX_PROCESSES 12
#define MAX_THREADS 10
#define NUM_THREADS 3

typedef struct threadPackage_t
{

	char *filename;
	int startOffset;
	int endOffset;
	char **processList;
	int numProcesses;
	int *processCount;
	//int *totalLines;
};

void threadFunc(threadPackage_t *package)
{
	char **splitBuff;
	int i,numTokens;
	FILE *flog;
	char tempBuff[BUFFER_SIZE] = {'\0'};

	//Allocate the array which will be used to hold each individual word in a line.
	splitBuff = (char **) malloc ( MAX_WORDS * sizeof(char *));
	for(i = 0; i < MAX_WORDS;i++)
	{
		splitBuff[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}
	flog = fopen(package->filename,"r");
	if (flog == NULL)
	{
		pthread_exit(NULL);
	}


	//Read the log file line by line
	while (fgets(tempBuff,BUFFER_SIZE,flog) != NULL)
	{
		numTokens = splitLine(tempBuff,splitBuff," :[]");
		
		//Check if the current logLine is written by a process we're interested in.
		if ( (i = arrayContains(package->processList,splitBuff[5],package->numProcesses)) > 0)
		{
			package->processCount[i]++;
		}
		//for (i = 0; i < numTokens; i++)
		//{
		//	printf("\t%d: %s",i,splitBuff[i]);
		//}
		//(package->*totalLines)++;
	}
	fclose(flog);
	for (i = 0; i < package->numProcesses; i++)
	{
		free(splitBuff[i]);
	}
	free(splitBuff);
}

int adjustThreadOffsets(FILE *flog,int fileSize,int *offsetArray,int numThreads)
{
	int i,c, blockSize;

	blockSize = fileSize / numThreads;
	offsetArray[0] = 0;
	
	//Adjust the offsets so that each thread gets atleast blockSize + EOL to process.
	for (i = 1; i < numThreads; i++)
	{
		fseek(flog,blockSize,offsetArray[i - 1]);
		while ( (c = fgetc(flog)) != '\n' || (c = fgetc(flog)) != EOF )
		{
			if ( c == '\n')
			{
				offsetArray[i] = ftell(flog);
			}
			else if (c == EOF)
			{
				//TODO: go back if EOF is hit
				//TODO: check if this can happen more than once.
				offsetArray[i] = -1;
			}
		}
		
	}

}

int main(int argc, char **argv)
{
	FILE *flog,*fproc;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	int numTokens,i = 0,j = 0,numProcesses,totalLines = 0,fileSize,threadLines;
	char *processList[MAX_PROCESSES];
	int processCount[MAX_THREADS][MAX_PROCESSES],offsetArray[MAX_THREADS];
	struct timespec start_time,end_time;
	pthread_t threads[MAX_THREADS];
	threadPackage_t threadPackage;

	if (argc != 3)
	{
		printf("\nIncorrect usage!\n Usage - %s path-to-log-file path-to-process-list-file.\n",argv[0]);
		return -1;
	}

	clock_gettime(CLOCK_MONOTONIC,&start_time);

	fproc = fopen(argv[2],"r");
	if (fproc != NULL)
	{
		printf("\nError opening processfile!\n");
		return -1;
	}
	for (i = 0; i < MAX_THREADS; i++)
	{
		for (j = 0; i < MAX_PROCESSES; j++)
		{
			processCount[i][j] = 0;
		}
	}
	
	//Read every processName from the processes file
	while (fscanf(fproc,"%s",tempBuff) >= 0)
	{
		processList[i] = strdup(tempBuff);
		i++;
	}
	fclose(fproc);
	numProcesses = i;
	
	flog = fopen(argv[1],"r");
	if (flog != NULL)
	{
		printf("\nError opening logfile!\n");
		return -1;
	}
	//Get the total byte size of the file.
	//fileSize = fseek(flog,0,SEEK_END);
	//adjustThreadOffsets(flog,fileSize,offsetArray,NUM_THREADS);
	//fclose(flog);
	
	//Get the linecount
	while ( fgets(tempBuff,1024,flog) > 0)
	{
		totalLines++;
	}
	threadLines = totalLines / NUM_THREADS;
	offsetArray[0] = 0;
	for ( i = 1; i < NUM_THREADS; i++)
	{
		offsetArray[i] = offsetArray[i - 1] + threadLines;
	}

	threadPackage.processList = processList;
	threadPackage.numProcesses = numProcesses;
	strcpy(threadPackage.filename,argv[1]);

	for (i = 0 ;i < NUM_THREADS; i++)
	{
		threadPackage.processCount = processCount[i];

		threadPackage.startOffset = offsetArray[i];
		if (i != NUM_THREADS - 1)
		{
			threadPackage.endOffset = offsetArray[i+1] - 1;
		}
		else
		{
			threadPackage.endOffset = - 1;
		}
		pthread_create(&threads[i],NULL,threadFunc,&threadPackage);
	}

	for (i = 0 ;i < NUM_THREADS; i++)
	{
		if (pthread_join(thread[i],NULL) != 0)
		{
			printf("\nERROR: pthread_join returned non-zero!\n");
		}
	}

	//Now collate all the process counts received from each threads into one.
	for (i = 0 ; i < NUM_THREADS; i++)
	{
		for (j = 0; j < MAX_PROCESSES; j++)
		{
			processCount[NUM_THREADS][j] += processCount[i][j];
		}
		totalLines += processCount[i][j];
	}

	clock_gettime(CLOCK_MONOTONIC,&start_time);

	printf("\n%d: %d: %d: %lf: ",totalLines,1,totalLines,(end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0);
	for(i = 0; i < numProcesses;i++)
	{
		printf("\npName:%s count:%d",processList[i],processCount[NUM_THREADS][i]);
	}
	printf("\nTotal Number of loglines: %d\n",totalLines);

	//Be a good boy and release all the memory
	for (i = 0; i < numProcesses; i++)
	{
		free(processList);
	}
	return 0;

}
