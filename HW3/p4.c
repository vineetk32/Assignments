#include "p4.h"

#define DEBUG

#ifdef _WIN32
#define __func__ __FUNCTION__
#endif

int systemLogLevel;

int main(int argc, char **argv)
{
	FILE *fCountry,*fFileList,*fMovie;
	List_t countryList;
	List_t movieList;
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	char **splitBuff;
	int i;
	MovieTuple_t tempTuple;
	myHashEntry_t tempEntry;

#ifdef DEBUG
	systemLogLevel = VDEBUG;
#else
	systemLogLevel = VSEVERE;
#endif

	initList(&countryList);
	initList(&movieList);

	if (argc != 3)
	{
		printf("Incorrect usage!\nUsage - %s path-to-countries-file path-to-file-of-files.\n",argv[0]);
		return -1;
	}

	fCountry = fopen(argv[1],"r");
	if (fCountry == NULL)
	{
		writeLog(__func__,VSEVERE,systemLogLevel,"Error opening the countries file! (%s)\n",argv[1]);
		return -1;
	}

	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fCountry) != NULL)
	{
		tempBuff[strlen(tempBuff) - 1] = '\0';
		addToList(&countryList,tempBuff,strlen(tempBuff)+1);
		tempBuff[0] = '\0';
	}
	
#ifdef DEBUG
	printf("Countries - ");
	printList(&countryList);
#endif

	fclose(fCountry);

	fFileList = fopen(argv[2],"r");
	if (fFileList == NULL)
	{
		writeLog(__func__,VSEVERE,systemLogLevel,"Error opening the File-of-Files! (%s)\n",argv[2]);
		return -1;
	}

	splitBuff = (char **) malloc(sizeof(char *) * 8);
	
	for(i = 0; i < 8;i++)
	{
		splitBuff[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}
	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fFileList) != NULL)
	{
		tempBuff[strlen(tempBuff) - 1] = '\0';
		fMovie = fopen(tempBuff,"r");
		if (fMovie == NULL)
		{
			writeLog(__func__,VSEVERE,systemLogLevel,"Error opening movie file! (%s)",tempBuff);
		}
		else
		{
#ifdef DEBUG
			printf("Now reading from %s\n",tempBuff);
#endif
			tempBuff[0] = '\0';
			while (fgets(tempBuff,SMALL_BUFFER_SIZE,fMovie) != NULL)
			{
				tempBuff[strlen(tempBuff) -1 ] = '\0';
				if (strlen(tempBuff) > 5)
				{
#ifdef DEBUG
					printf("Splitting %s\n",tempBuff);
#endif
					if (splitLine(tempBuff,splitBuff,":") != 5)
					{
						writeLog(__func__,VERROR,systemLogLevel,"Badly formed movie record: %s",tempBuff);
					}
					else
					{
						strcpy(tempTuple.movieName,splitBuff[0]);
						tempTuple.movieVotes = (unsigned int) atoi(splitBuff[1]);
						tempTuple.movieRating = (unsigned short) atoi(splitBuff[2]);
						tempTuple.movieYear = (unsigned short) atoi(splitBuff[3]);
						strcpy(tempTuple.movieCountry,splitBuff[4]);
					}
					tempEntry.keys = (char **) malloc (sizeof(char *) * 2);
					tempEntry.keys[0] = 
					addToList(&movieList,&tempTuple,sizeof(tempTuple));
				}
				tempBuff[0] = '\0';
			}
		tempBuff[0] = '\0';
		fclose(fMovie);
		}
	}

	for (i = 0; i < 8; i++)
	{
		free(splitBuff[i]);
	}
	free(splitBuff);

	return 0;
}

int addToHashTable(myHashTable_t *table, myHashEntry_t *entry)
{
	//TODO: Handle collisions later.
	int i = 0;
	unsigned short bucketIndex = 0;
	for (i = 0; i < entry->numKeys; i++)
	{
		bucketIndex = hashFunction(bucketIndex,entry->keys[i],BUCKET_SIZE);
	}
	if (table->entries[bucketIndex].numKeys != 0)
	{
		writeLog(__func__,VINFO,systemLogLevel,"Collision for keys %s and %s.",entry->keys[0],entry->keys[1]);
	}
	memcpy(&table->entries[bucketIndex],entry,sizeof(myHashEntry_t));
}
int getFromHashTable(myHashTable_t *table, myHashEntry_t *entry)
{
	int i = 0;
	unsigned short bucketIndex = 0;
	for (i = 0; i < entry->numKeys; i++)
	{
		bucketIndex = hashFunction(bucketIndex,entry->keys[i],BUCKET_SIZE);
	}
	if (table->entries[bucketIndex].numKeys == 0)
	{
		writeLog(__func__,VINFO,systemLogLevel,"No entry for keys %s and %s.",entry->keys[0],entry->keys[1]);
	}
	else
	{
		entry = &table->entries[bucketIndex];
	}
}

//Simple CheckSum mod BucketSize hash.
int hashFunction(int seed,char *key,int bucketSize)
{
	int tempIndex, i;
	tempIndex = seed;
	for (i = 0; i < strlen(key); i++)
	{
		tempIndex += key[i];
	}
	tempIndex = tempIndex % bucketSize;
	return tempIndex;
}

void printHashTable(myHashTable_t *table)
{
	int i = 0;
	printf("\n=========================================");
	for(i = 0; i < BUCKET_SIZE; i++)
	{
		if (table->entries[i].numKeys > 0)
		{
			printf("\n[%u]",i);
			for (j = 0; j < table->entries[i].numKeys; j++)
			{
				printf("\t%s",table->entries[i].keys[j]);
			}
		}
	}
	printf("\n=========================================");
}
