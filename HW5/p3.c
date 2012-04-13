#include "p3.h"

int systemLogLevel;

#define DEBUG
int totalFiles,totalWords;

int main(int argc, char **argv)
{
	char tempBuff[BUFFER_SIZE] = {'\0'};
	myHashTable_t hashTable;
	collidedEntry_t *collisions;
	int numCollisions = 0, systemLogLevel;
	int i;
	FILE *fCorpus,*fFileList,*fdataFile;
	int numWords,fileSize;
	char **corpusWords;
	char currToken[SMALL_BUFFER_SIZE] = {'\0'};
	char *fileContents;

#ifndef _WIN32
		struct timespec start_time,end_time;
		clock_gettime(CLOCK_MONOTONIC,&start_time);
#endif

	initHashTable(&hashTable);

	collisions = (collidedEntry_t *) malloc (sizeof(collidedEntry_t) * 512);
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
		addToHashTable(&hashTable,corpusWords[i],collisions,&numCollisions);
		i++;
	}
	numWords = i;
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


	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fFileList) != NULL)
	{
		totalFiles++;
		tempBuff[strlen(tempBuff) -1] = '\0';
		fdataFile = fopen(tempBuff,"r");
		if (fdataFile == NULL)
		{
			printf("\nError opening datafile!\n");
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

		actualWorkFunction(fileContents,0,fileSize,&hashTable,collisions,&numCollisions,corpusWords,numWords);
		free(fileContents);
	}

#ifndef _WIN32
		clock_gettime(CLOCK_MONOTONIC,&end_time);
#endif

#ifdef DEBUG
	printf("\nHash Table contents -  ");
	printHashTable(&hashTable);
	printf("\nCollided Entries - ");
	printf("\n=========================================");
	for (i = 0; i < numCollisions; i++)
	{
		printf("\n%d (%d) %s : %d",i,collisions[i].bucketIndex,collisions[i].entry.key,collisions[i].entry.value);
	}
	printf("\n=========================================");
#endif

	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (hashTable.entries[i].value != 0)
		{
			printf("\n%s:%d:4",hashTable.entries[i].key,--hashTable.entries[i].value);
		}
	}
	printf("\n%d:%d:",totalFiles,totalWords);
	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (hashTable.entries[i].value != 0)
		{
			printf("\n%s:%f",hashTable.entries[i].key,(float) hashTable.entries[i].value/totalWords);
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

void actualWorkFunction(char *dataBuff,int start,int end,myHashTable_t *table, collidedEntry_t *collisions, int *numCollisions,char **corpusWords,int numWords)
{
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	int i, j = 0;
	
	//Skip till you reach a whitespace char
	//while (dataBuff[start] != ' ')
	//{
	//	if (dataBuff[start] != '\0')
	//	{
	//		start++;
	//	}
	//}
	
	tempBuff[0] = '\0';
	//printf("\n");
	for (i = start; i < end; i++)
	{
		if ( dataBuff[i] == ' ' || i == end - 1)
		{
			j = 0;

//#ifdef DEBUG
//			printf("(%s)",tempBuff);
//#endif
			if (arrayContains(corpusWords,tempBuff,numWords) >= 0)
			{
				addToHashTable(table,tempBuff,collisions,numCollisions);
			}
			memset(tempBuff,0,MEDIUM_BUFFER_SIZE);
			totalWords++;
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
