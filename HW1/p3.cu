#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "util.h"
#include <pthread.h>

#define MAX_THREADS 20
#define NUM_THREADS 8
#define MAX_PROCESSES 12

struct threadPackage_t
{
	long startOffset;
	long endOffset;
	char **processList;
	int numProcesses;
	int *processCount;
	char *buffer;
};

void *threadFunc(void *threadPackage)
{
	char **splitBuff;
	int i,numTokens;

	struct threadPackage_t *package;
	char *tempBuff;

	long start,end,curr;
	int exitFlag = 0;

	package = (struct threadPackage_t *) threadPackage; 

	start = package->startOffset;
	end = package->endOffset;


	#ifdef __DEBUG
	printf("\nstart: %ld end: %ld curr: %ld - ",start,end,curr);
	#endif

	//Allocate the array which will be used to hold each individual word in a line.
	splitBuff = (char **) malloc ( MAX_WORDS * sizeof(char *));
	tempBuff = (char *) malloc ( BUFFER_SIZE * sizeof(char));

	for(i = 0; i < MAX_WORDS;i++)
	{
		splitBuff[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}

	//Read the input data between the offsets char-by-char
	while (exitFlag == 0)
	{
		/*while (package->buffer[start] != '\n')
		{
			start++;
			if (package->buffer[start] == '\0')
			{
				exitFlag = 1;
				break;
			}
		}*/
		start++;
		curr = start + 1;
		if (end != -1)
		{
			while (package->buffer[curr] != '\n' && curr < end)
			{
				curr++;
			}

			if (curr >= end)
			{
				exitFlag = 1;
			}
			strncpy(tempBuff,(package->buffer) + start,curr-start);
			curr++;
		}
		else
		{
			while (package->buffer[curr] != '\n' && package->buffer[curr] != '\0')
			{
				curr++;
			}

			if (package->buffer[curr] == '\0')
			{
				exitFlag = 1;
			}
			strncpy(tempBuff,(package->buffer)+ start,curr-start);
		}
		#ifdef __DEBUG
		printf("\nReading %d --> %d (%s)",start,curr,tempBuff);
		#endif
		tempBuff[curr - start - 1] = '\0';
		start = curr;
		if (strlen(tempBuff) == 0)
		{
			#ifdef __DEBUG
			printf("\nZero-len tempBuff! start - %ld curr - %ld end - %ld",start,curr,end);
			#endif
		}
		else
		{
			package->processCount[package->numProcesses]++;
		}

		#ifdef __DEBUG
		//printf("\nTempBuff - %s",tempBuff);
		#endif
	
		//TODO: Apparently, strtok_r is thread-safe only for dynamic mem. Check
		numTokens = splitLine(tempBuff,splitBuff," :[]");
		
		//Check if the current logLine is written by a process we're interested in.
		if (numTokens > 5)
		{
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
		else
		{
			#ifdef __DEBUG
			printf("\nBad Line - %s",tempBuff);
			#endif
		}
	}
	for (i = 0; i < package->numProcesses; i++)
	{
		free(splitBuff[i]);
	}
	free(splitBuff);
}

int adjustThreadOffsets(char *fileContents,int fileSize,long *offsetArray,int numThreads)
{
	int i,blockSize;
	long ptr = 0;

	blockSize = fileSize / numThreads;
	offsetArray[0] = 0;
	
	//Adjust the offsets so that each thread gets atleast blockSize + EOL to process.
	for (i = 1; i < numThreads; i++)
	{
		//If previous offset is set to EOF, all further offsets must also be set to EOF.
		if (offsetArray[i - 1] != -1)
		{
			//fseek(flog,blockSize,offsetArray[i - 1]);
			ptr += blockSize;
		}
		else
		{
			offsetArray[i] = -1;
		}
		while ( fileContents[ptr] != '\n' && fileContents[ptr] != '\0')
		{
			ptr++;
		}
		
		if ( fileContents[ptr] == '\n')
		{
			offsetArray[i] = ptr;
		}
		else if (fileContents[ptr] == '\0')
		{
			offsetArray[i] = -1;
		}
	}

	#ifdef __DEBUG
	printf("\noffsetArray - ");
	for (i = 0; i < numThreads; i++)
	{
		printf(" %d: %ld",i,offsetArray[i]);
	}
	#endif
}

int main(int argc, char **argv)
{
	FILE *flog,*fproc;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	int i = 0,j = 0,k = 0,numProcesses,totalLines = 0,fileSize;
	char *processList[MAX_PROCESSES];
	int processCount[MAX_THREADS][MAX_PROCESSES];
	long offsetArray[MAX_THREADS];
	struct timespec start_time,end_time;
	pthread_t threads[MAX_THREADS];
	char *fileContents;
	struct threadPackage_t threadPackage[MAX_THREADS];
	int printedOnce = 0,numThreads;


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
	//Get the total byte size of the file.
	if (fseek(flog,0,SEEK_END) == 0)
	{
		fileSize = ftell(flog);
	}

	fileContents = (char *) malloc (sizeof(char) * fileSize);
	rewind(flog);
	fread(fileContents,1,fileSize,flog);
	fclose(flog);

	for (k = 1; (k * 2) <  16; k++)
	{
		numThreads = k * 2;
		for (i = 0; i <= numThreads; i++)
		{
			for (j = 0; j < MAX_PROCESSES; j++)
			{
				processCount[i][j] = 0;
			}
		}

		adjustThreadOffsets(fileContents,fileSize,offsetArray,numThreads);

		//Get the linecount
		/*while ( fgets(tempBuff,1024,flog) > 0)
		{
			totalLines++;
		}
		threadLines = totalLines / NUM_THREADS;
		offsetArray[0] = 0;
		for ( i = 1; i < NUM_THREADS; i++)
		{
			offsetArray[i] = offsetArray[i - 1] + threadLines;
		}*/

		for (i = 0 ;i < numThreads; i++)
		{
			threadPackage[i].processCount = processCount[i];
			threadPackage[i].processList = processList;
			threadPackage[i].numProcesses = numProcesses;
			threadPackage[i].buffer = fileContents;
			threadPackage[i].startOffset = offsetArray[i];

			if (i < (numThreads - 1))
			{
				threadPackage[i].endOffset = offsetArray[i+1] - 1;
			}
			else
			{
				threadPackage[i].endOffset = -1;
			}

			#ifdef __DEBUG
			printf("\nThreadPackage[%d] - \n",i);
			printf("startOffset: %ld endOffset: %ld\n",threadPackage[i].startOffset,threadPackage[i].endOffset);
			#endif
			pthread_create(&threads[i],NULL,&threadFunc,&threadPackage[i]);
		}

		for (i = 0 ;i < numThreads; i++)
		{
			if (pthread_join(threads[i],NULL) != 0)
			{
				printf("\nERROR: pthread_join returned non-zero!\n");
			}
		}

		clock_gettime(CLOCK_MONOTONIC,&end_time);
		//blockSize: numOfThreads: totalCount: runningTime: 

		//Now collate all the process counts received from each threads into one.
		for (i = 0 ; i < numThreads; i++)
		{
			for (j = 0; j < MAX_PROCESSES; j++)
			{
				processCount[numThreads][j] += processCount[i][j];
			}
			totalLines += processCount[i][numProcesses];
		}

		if (printedOnce == 0)
		{
			for(i = 0; i < numProcesses;i++)
			{
				printf("\npName:%s count:%d",processList[i],processCount[numThreads][i]);
			}
			printf("\nTotal Number of loglines: %d\n",totalLines);
			printedOnce = 1;
		}
		//printf("\nNow %d\n",numThreads);
		printf("\n%d: %d: %d: %lfs: \n",fileSize/numThreads,numThreads,totalLines,(end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0);
	}
	//Be a good boy and release all the memory
	for (i = 0; i < numProcesses; i++)
	{
		free(processList[i]);
	}
	return 0;

}
