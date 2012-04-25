#include "p3.h"

int systemLogLevel;
#define NUM_THREADS 1
//#define DEBUG
int totalFiles,totalWords;
int *changedThisTime;


void *threadFunc(void *arg)
{
	threadPackage_t *package = (threadPackage_t *) arg;
	actualWorkFunction(package->dataBuff,package->start,package->end,package->corpusTable,package->corpusCollisions,package->numCorpusCollisions,package->corpusWords,package->numWords,package->fileTable,package->fileCollisions,package->numFileCollisions,package->mutex);
}

int main(int argc, char **argv)
{
	char tempBuff[BUFFER_SIZE] = {'\0'};
	myHashTable_t wordCount,fileCount;
	collidedEntry_t *wordCollisions,*fileCollisions;
	int numWordCollisions = 0, systemLogLevel, numFileCollisions = 0;
	int i, fileSplit;
	FILE *fCorpus,*fFileList,*fdataFile;
	int numWords,fileSize;
	char **corpusWords;
	char currToken[SMALL_BUFFER_SIZE] = {'\0'};
	char *fileContents;
	pthread_t threads[NUM_THREADS];
	threadPackage_t threadPackages[NUM_THREADS];
	pthread_mutex_t mutex;

#ifndef _WIN32
		struct timespec start_time,end_time;
		clock_gettime(CLOCK_MONOTONIC,&start_time);
#endif

	initHashTable(&wordCount);
	initHashTable(&fileCount);

	//TODO: Realloc as needed
	wordCollisions = (collidedEntry_t *) malloc (sizeof(collidedEntry_t) * 512);
	fileCollisions = (collidedEntry_t *) malloc (sizeof(collidedEntry_t) * 512);

	corpusWords = (char **) malloc (sizeof(char *) * MAX_CORPUS_WORDS);
	

	for (i = 0; i < MAX_CORPUS_WORDS; i++)
	{
		corpusWords[i] = (char *) malloc (sizeof(char) * MAX_CORPUS_WORD_SIZE);
	}

#ifdef DEBUG
	systemLogLevel = VDEBUG;
#else
	systemLogLevel = VSEVERE;
#endif

	if (argc != 3)
	{
		printf("Incorrect usage!\nUsage - %s path-to-file-of-files path-to-corpus-file.\n",argv[0]);
		return -1;
	}

	fCorpus = fopen(argv[2],"r");
	if (fCorpus == NULL)
	{
		writeLog(__func__,VSEVERE,systemLogLevel,"Error opening the corpus file! (%s)\n",argv[1]);
		return -1;
	}
	i = 0;
	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fCorpus) != NULL)
	{
		tempBuff[strlen(tempBuff) - 1] = '\0';
		//addToList(&countryList,tempBuff,strlen(tempBuff)+1);
		strcpy(corpusWords[i],tempBuff);
		tempBuff[0] = '\0';
		addToHashTable(&wordCount,corpusWords[i],wordCollisions,&numWordCollisions);
		i++;
	}
	numWords = i;

	changedThisTime = (int *) malloc (sizeof(int) * numWords);

#ifdef DEBUG
	printf("Corpus-Words - ");
	for (i = 0; i < numWords; i++)
	{
		printf("\t%s",corpusWords[i]);
	}
	printf("\n");
#endif

	fclose(fCorpus);

	fFileList = fopen(argv[1],"r");
	if (fFileList == NULL)
	{
		writeLog(__func__,VSEVERE,systemLogLevel,"Error opening the File-of-Files! (%s)\n",argv[1]);
		return -1;
	}

	i = 0;
	
	pthread_mutex_init(&mutex,NULL);

	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fFileList) != NULL)
	{
		memset(changedThisTime,0,numWords * sizeof(int));
		totalFiles++;
		tempBuff[strlen(tempBuff) - 1] = '\0';
		fdataFile = fopen(tempBuff,"r");
		if (fdataFile == NULL)
		{
			printf("\nError opening datafile - %s \n",tempBuff);
			return -1;
		}

		if (fseek(fdataFile,0,SEEK_END) == 0)
		{
			fileSize = ftell(fdataFile);
		}

		fileContents = (char *) malloc (sizeof(char) * fileSize);
		rewind(fdataFile);
		fread(fileContents,1,fileSize,fdataFile);
		fclose(fdataFile);

		//actualWorkFunction(fileContents,0,fileSize,&wordCount,wordCollisions,&numWordCollisions,corpusWords,numWords,&fileCount,fileCollisions,&numFileCollisions);
		//void actualWorkFunction(char *dataBuff,int start,int end,myHashTable_t *table, collidedEntry_t *collisions, int *numCollisions,char **corpusWords,int numWords,myHashTable_t *fileHash,collidedEntry_t *fileCollisions, int *numFileCollisions);
		for (i = 0; i < NUM_THREADS; i++)
		{
			threadPackages[i].dataBuff = fileContents;
			threadPackages[i].corpusTable = &wordCount;
			threadPackages[i].corpusWords = corpusWords;
			threadPackages[i].fileCollisions = fileCollisions;
			threadPackages[i].fileTable = &fileCount;
			threadPackages[i].numCorpusCollisions = &numWordCollisions;
			threadPackages[i].numFileCollisions = &numFileCollisions;
			threadPackages[i].numWords = numWords;
			threadPackages[i].mutex = &mutex;

			if (i > 0)
			{
				threadPackages[i].start = threadPackages[i - 1].end + 1;
			}
			else
			{
				threadPackages[i].start = 0;
			}
			if ( i == (NUM_THREADS - 1))
			{
				threadPackages[i].end = fileSize;
			}
			else
			{
				threadPackages[i].end = threadPackages[i].start + fileSize / NUM_THREADS;
			}
//#ifdef DEBUG
			printf("\nThread %d: Start: %d, End %d, fileSize - %d",i,threadPackages[i].start,threadPackages[i].end,fileSize);
//#endif

			pthread_create(&threads[i],NULL,threadFunc,((void *) &threadPackages[i]));
		}

		for (i = 0; i < NUM_THREADS; i++)
		{
			pthread_join(threads[i],NULL);
		}
		free(fileContents);
	}

	pthread_mutex_destroy(&mutex);

#ifndef _WIN32
		clock_gettime(CLOCK_MONOTONIC,&end_time);
#endif

#ifdef DEBUG
	printf("\nHash Table contents -  ");
	printHashTable(&wordCount);
	printf("\nCollided Entries - ");
	printf("\n=========================================");
	for (i = 0; i < numWordCollisions; i++)
	{
		printf("\n%d (%d) %s : %d",i,wordCollisions[i].bucketIndex,wordCollisions[i].entry.key,wordCollisions[i].entry.value);
	}
	printf("\n=========================================");
#endif

	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (wordCount.entries[i].value != 0)
		{
			printf("\n%s:%d:%d",wordCount.entries[i].key,--wordCount.entries[i].value,fileCount.entries[i].value);
		}
	}
	printf("\n%d:%d:",totalFiles,totalWords);
	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (wordCount.entries[i].value != 0)
		{
			printf("\n%s:%f",wordCount.entries[i].key,(float) wordCount.entries[i].value/totalWords);
		}
	}

#ifndef _WIN32
	printf("\nCPUTime:%lfs",(end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_nsec - start_time.tv_nsec)/1000000000.0);
#else
	printf("\nCPUTime:");
#endif
	printf("\nGPUTime:");
	return 0;
}

void initHashTable(myHashTable_t *table)
{
	int i;
	for (i = 0; i < BUCKET_SIZE; i++)
	{
		table->entries[i].key[0] = '\0';
		table->entries[i].value = 0;
	}
}

int addToHashTable(myHashTable_t *table, char *key,collidedEntry_t *collisions,int *numCollisions)
{
	int i = 0;
	unsigned int bucketIndex = 0;

	bucketIndex = hashFunction(key);
	bucketIndex = bucketIndex % BUCKET_SIZE;

	if (table->entries[bucketIndex].value != 0)
	{
		if (strcmp(key,table->entries[bucketIndex].key) != 0)
		{
			writeLog(__func__,VINFO,systemLogLevel,"Collision for keys %s%s and %s%s.",table->entries[bucketIndex].key,key);
			for (i = 0; i < *numCollisions; i++)
			{
				if ( strcmp(collisions[(*numCollisions)].entry.key,key) == 0)
				{
					collisions[(*numCollisions)].entry.value++;
					return 0;
				}
			}
			strcpy(collisions[(*numCollisions)].entry.key,key);
			collisions[(*numCollisions)].bucketIndex = bucketIndex;
			collisions[(*numCollisions)].entry.value = 1;
			(*numCollisions)++;
		}
		else
		{
			table->entries[bucketIndex].value++;
		}
		return 0;
	}
	strcpy(table->entries[bucketIndex].key,key);
	table->entries[bucketIndex].value = 1;
	return 0;
}

unsigned int getFromHashTable(myHashTable_t *table, char *key)
{
	int i = 0;
	unsigned int bucketIndex = 0;
	bucketIndex = hashFunction(key);
	bucketIndex = bucketIndex % BUCKET_SIZE;
	if (table->entries[bucketIndex].key == NULL)
	{
		writeLog(__func__,VINFO,systemLogLevel,"No entry for key %s.",key);
		return BUCKET_SIZE + 1;
		//return NULL;
	}
	else
	{
		//entry = &table->entries[bucketIndex];
		return bucketIndex;
	}
	//return entry;
}

//djb2 hash - taken from StackOverflow
unsigned long hashFunction(char *str)
{
	unsigned long hash = 5381;
	int c;

	while (c = *str++)
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

	return hash;
}

void actualWorkFunction(char *dataBuff,int start,int end,myHashTable_t *table, collidedEntry_t *collisions, int *numCollisions,char **corpusWords,int numWords,myHashTable_t *fileHash,collidedEntry_t *fileCollisions, int *numFileCollisions, pthread_mutex_t *mutex)
{
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	int i, j = 0;
	int wordIndex = 0;
	
	//Skip till you reach a whitespace char, unless at the very beginning of the buffer
	while (dataBuff[start] != ' ' && start < end && start != 0)
	{
		if (dataBuff[start] != '\0')
		{
			start++;
		}
	}
	
	tempBuff[0] = '\0';
	//printf("\n");
	for (i = start; i < end; i++)
	{
		if ( dataBuff[i] == ' ' || i == end - 1)
		{
			j = 0;
			if ( (wordIndex = arrayContains(corpusWords,tempBuff,numWords)) >= 0)
			{
#ifdef DEBUG
				printf("\nFound %s ending at %d\n",corpusWords[wordIndex],i);
#endif
				pthread_mutex_lock(mutex);
				addToHashTable(table,tempBuff,collisions,numCollisions);
				pthread_mutex_unlock(mutex);
				if (changedThisTime[wordIndex] == 0)
				{
					//TODO: Add a collision set for fileHash
					pthread_mutex_lock(mutex);
					addToHashTable(fileHash,tempBuff,fileCollisions,numFileCollisions);
					pthread_mutex_unlock(mutex);
					changedThisTime[wordIndex] = 1;
				}
			}
			memset(tempBuff,0,MEDIUM_BUFFER_SIZE);
			pthread_mutex_lock(mutex);
			totalWords++;
			pthread_mutex_unlock(mutex);
		}
		else if (dataBuff[i] != '\n')
		{
			tempBuff[j++] = dataBuff[i];
		}
	}
}

void printHashTable(myHashTable_t *table)
{
	int i = 0,j = 0;
	printf("\n=========================================");
	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (table->entries[i].value != 0)
		{
			printf("\n[%u] %s : %d",i,table->entries[i].key,table->entries[i].value);
		}
	}
	printf("\n=========================================");
}
