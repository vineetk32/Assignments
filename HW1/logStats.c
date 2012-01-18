#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256
#define BUFFER_SIZE 1024
#define MAX_PROCESSES 20

int splitLine(const char *in, char **out,const char *delim)
{
	int i = 0;
	char *ptr, *saveptr;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	strcpy(tempBuff,in);

	ptr = strtok_r(tempBuff,delim,&saveptr);
	while (ptr != NULL)
	{
		out[i][0] = '\0';
		strcpy(out[i],ptr);
		i++;
		ptr = strtok_r(NULL,delim,&saveptr);

	}
	return i;

}

int arrayContains(const char **array, const char *element, int arrayLen)
{
	int i;
	for (i = 0; i < arrayLen; i++)
	{
		if (strcmp(array[i],element) == 0)
		{
			return i;
		}
	}
	return -1;
}

int main(int argc, char **argv)
{
	FILE *flog,*fproc;
	char tempBuff[BUFFER_SIZE] = {'\0'} ,**splitBuff;
	int numTokens,i = 0,numProcesses,totalLines = 0;
	char *processList[MAX_PROCESSES];
	int processCount[MAX_PROCESSES] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	struct timespec start_time,end_time;

	if (argc != 3)
	{
		printf("\nIncorrect usage!\n Usage - %s path-to-log-file path-to-process-list-file.\n",argv[0]);
		return -1;
	}

	flog = fopen(argv[1],"r");
	fproc = fopen(argv[2],"r");

	clock_gettime(CLOCK_MONOTONIC,&start_time);
	if (fproc != NULL)
	{
		printf("\nError opening processfile!\n");
		return -1;
	}
	
	while (fscanf(fproc,"%s",tempBuff) >= 0)
	{
		processList[i] = strdup(tempBuff);
		i++;
	}
	numProcesses = i;
	if (flog != NULL)
	{
		printf("\nError opening logfile!\n");
		return -1;
	}
	splitBuff = (char **) malloc ( MAX_WORDS * sizeof(char *));
	for(i = 0; i < MAX_WORDS;i++)
	{
		splitBuff[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}

	while (fgets(tempBuff,BUFFER_SIZE,flog) != NULL)
	{
		numTokens = splitLine(tempBuff,splitBuff," :[]");
		if ( (i = arrayContains(processList,splitBuff[5],numProcesses)) > 0)
		{
			processCount[i]++;	
		}
		//for (i = 0; i < numTokens; i++)
		//{
		//	printf("\t%d: %s",i,splitBuff[i]);
		//}
		totalLines++;
	}
	fclose(flog);
	clock_gettime(CLOCK_MONOTONIC,&start_time);

	printf("\n%d: %d: %d: %lf: ",totalLines,1,totalLines,(end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0);
	for(i = 0; i < numProcesses;i++)
	{
		printf("\npName:%s count:%d",processList[i],processCount[i]);
	}
	printf("\nTotal Number of loglines: %d\n",totalLines);
	return 0;

}
