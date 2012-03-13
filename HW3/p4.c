#include "p4.h"

//#define DEBUG

#ifdef _WIN32
#define __func__ __FUNCTION__
#endif

#define MAX_COUNTRIES 8
#define CHARS_PER_LINE 128
#define LINES_PER_COUNTRY 10000

//Yes, I made all of them global
int systemLogLevel;
char **countryArray;
short countryYearArray[MAX_COUNTRIES][64];
short countryReleasesInYear[MAX_COUNTRIES][64];
int countryNumYears[MAX_COUNTRIES];
int countryNumMovies[MAX_COUNTRIES];

int numCountries = 0,numYears = 0;
short globalYearArray[64];
short globalReleasesInYear[64];
myHashTable_t hashTable;

collidedEntry_t collisions[256];
int numCollisions = 0;


int main(int argc, char **argv)
{
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};

	int i,j,k;

	FILE *fCountry,*fFileList,*fMovie;
	MovieTuple_t *highestRatedMovie;
	myHashEntry_t *tempEntry;

	int index;
	unsigned short highestRating = 0,thisRating = 0;
	char ***dataBuff;
	char **tempKeys,**splitBuff;

	
	dataBuff = malloc(sizeof(char **) * MAX_COUNTRIES);
	countryArray = malloc(sizeof(char *) * MAX_COUNTRIES);
	for (i = 0; i< MAX_COUNTRIES; i++)
	{
		dataBuff[i] = malloc(sizeof(char *) * LINES_PER_COUNTRY);
		countryArray[i] = malloc(sizeof(char) * SMALL_BUFFER_SIZE);
		countryNumYears[i] = 0;
		countryNumMovies[i] = 0;
		for (j = 0; j < LINES_PER_COUNTRY; j++)
		{
			dataBuff[i][j] = malloc (sizeof(char) * CHARS_PER_LINE);
		}
	}

	initHashTable(&hashTable);

#ifdef DEBUG
	systemLogLevel = VDEBUG;
#else
	systemLogLevel = VSEVERE;
#endif

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
	i = 0;
	while (fgets(tempBuff,SMALL_BUFFER_SIZE,fCountry) != NULL)
	{
		tempBuff[strlen(tempBuff) - 1] = '\0';
		//addToList(&countryList,tempBuff,strlen(tempBuff)+1);
		strcpy(countryArray[i],tempBuff);
		tempBuff[0] = '\0';
		i++;
	}
	numCountries = i;
#ifdef DEBUG
	printf("Countries - ");
	for (i = 0; i < numCountries; i++)
	{
		printf("\t%s",countryArray[i]);
	}
	printf("\n");
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
	i = 0;


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
						//Nice! so I will split the line, an put the unsplit line back in the buffer!
						index = arrayContains(countryArray,splitBuff[4],numCountries);
						strcpy(dataBuff[index][countryNumMovies[index]++],tempBuff);
					}
				}
				tempBuff[0] = '\0';
			}
		tempBuff[0] = '\0';
		fclose(fMovie);
		}
	}

	for (i = 0; i < numCountries;i++)
	{
		actualWorkFunction(dataBuff[i],0,countryNumMovies[i]);
	}

#ifdef DEBUG
	printf("Years - ");
	for (i = 0; i < numYears; i++)
	{
		printf("\t%d",globalYearArray[i]);
	}
	printf("\n");
#endif

	tempKeys = malloc(sizeof(char *) * 2);
	tempKeys[0] = malloc(sizeof(char) * 8);
	tempKeys[1] = malloc(sizeof(char) * 64);

	//Output Phase1 - top rated movies per year, per country
	for (i = 0; i < numCountries; i++)
	{
		printf("\n\n%s:",countryArray[i]);
		sortYearReleases(countryYearArray[i],countryReleasesInYear[i],countryNumYears[i]);

		for (j = 0; j < countryNumYears[i]; j++)
		{
			printf("\n%d:%d",countryYearArray[i][j],countryReleasesInYear[i][j]);

			sprintf(tempKeys[0],"%d",countryYearArray[i][j]);
			strcpy(tempKeys[1],countryArray[i]);
			index = getFromHashTable(&hashTable,tempKeys,2);
			if (index > BUCKET_SIZE)
			{
				continue;
			}
			else
			{
				tempEntry = &(hashTable.entries[index]);
				printf("\n%s",((MovieTuple_t *)tempEntry->ptr)->line);
				//Check collided Entries also
				for (k = 0 ; k < numCollisions;k++)
				{
					if (collisions[k].bucketIndex == index)
					{
						printf("\n%s",collisions[k].tuple.line);
					}
				}

			}
		}
	}
	printf("\n");
	sortYearReleases(globalYearArray,globalReleasesInYear,numYears);
	
	//Output Phase2 - top rated movies per year across countries.
	for(i = 0; i < numYears;i++)
	{
		sprintf(tempKeys[0],"%d",globalYearArray[i]);
		//tempEntry->keys[0] = tempBuff;
		highestRating = 0;
		for(j = 0; j < numCountries;j++)
		{
			strcpy(tempKeys[1],countryArray[j]);

			index = getFromHashTable(&hashTable,tempKeys,2);
			if (index < BUCKET_SIZE)
			{
				tempEntry = &(hashTable.entries[index]);
				if ( ((MovieTuple_t *)tempEntry->ptr)->movieRating > highestRating)
				{
					highestRatedMovie = ((MovieTuple_t *)tempEntry->ptr);
					highestRating = highestRatedMovie->movieRating;
				}
				else if ( ((MovieTuple_t *)tempEntry->ptr)->movieRating == highestRating)
				{
					if ( ((MovieTuple_t *)tempEntry->ptr)->movieVotes > highestRatedMovie->movieVotes)
					{
						highestRatedMovie = ((MovieTuple_t *)tempEntry->ptr);
					}
				}
			}
		}
		printf("\n%d:%d:",globalYearArray[i],globalReleasesInYear[i]);
		//printf("\n%s:%d:%d:%s",highestRatedMovie->movieName,highestRatedMovie->movieVotes,highestRatedMovie->movieRating,highestRatedMovie->movieCountry);
		printf("\n%s",highestRatedMovie->line);
		//Check collided Entries also
		for (k = 0 ; k < numCollisions;k++)
		{
			if (collisions[k].bucketIndex == index)
			{
				printf("\n%s",collisions[k].tuple.line);
			}
		}
	}
	printf("\n\n");
	for (i = 0; i < 8; i++)
	{
		free(splitBuff[i]);
	}
	free(splitBuff);
	for (i = 0; i < MAX_COUNTRIES; i++)
	{
		free(countryArray[i]);
	}
	free(countryArray);
	free(tempKeys[0]);
	free(tempKeys[1]);
	free(tempKeys);

	return 0;
}

void initHashTable(myHashTable_t *table)
{
	int i;
	for (i = 0; i < BUCKET_SIZE; i++)
	{
		table->entries[i].numKeys = 0;
	}
}

int addToHashTable(myHashTable_t *table, char **keys, int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions)
{
	int i = 0;
	unsigned int bucketIndex = 0;
	for (i = 0; i < numKeys; i++)
	{
		bucketIndex += hashFunction(bucketIndex,keys[i]);
	}
	bucketIndex = bucketIndex % BUCKET_SIZE;
	if (table->entries[bucketIndex].numKeys != 0)
	{
		writeLog(__func__,VINFO,systemLogLevel,"Collision for keys %s%s and %s%s.",table->entries[bucketIndex].keys[0],table->entries[bucketIndex].keys[1],keys[0],keys[1]);
		if ( collisionBreaker(&table->entries[bucketIndex],ptr) == 1)
		{
			memcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t));
		}
		else if ( collisionBreaker(&table->entries[bucketIndex],ptr) == -1)
		{
			memcpy(&collisions[(*numCollisions)].tuple,ptr,sizeof(MovieTuple_t));
			collisions[(*numCollisions)].bucketIndex = bucketIndex;
			(*numCollisions)++;
		}
		return 0;
	}
	table->entries[bucketIndex].numKeys = numKeys;
	table->entries[bucketIndex].keys = malloc(sizeof(char *) * 2);
	table->entries[bucketIndex].keys[0] = malloc (sizeof(char) * 8);
	table->entries[bucketIndex].keys[1] = malloc (sizeof(char) * 64);
	strcpy(table->entries[bucketIndex].keys[0],keys[0]);
	strcpy(table->entries[bucketIndex].keys[1],keys[1]);
	table->entries[bucketIndex].ptr = malloc(sizeof(MovieTuple_t));
	memcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t));
	return 0;
}

unsigned int getFromHashTable(myHashTable_t *table, char **keys, int numKeys)
{
	int i = 0;
	unsigned int bucketIndex = 0;
	for (i = 0; i < numKeys; i++)
	{
		bucketIndex += hashFunction(bucketIndex,keys[i]);
	}
	bucketIndex = bucketIndex % BUCKET_SIZE;
	if (table->entries[bucketIndex].numKeys == 0)
	{
		writeLog(__func__,VINFO,systemLogLevel,"No entry for keys %s and %s.",keys[0],keys[1]);
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

//Simple CheckSum mod BucketSize hash.
unsigned int hashFunction(unsigned int seed,char *key)
{
	int tempIndex, i;
	tempIndex = seed;

	for (i = 0; i < strlen(key); i++)
	{
		//tempIndex += key[i];
		tempIndex ^= (tempIndex << 5) + (tempIndex >> 2) + key[i];
	}
	//tempIndex = tempIndex % bucketSize;
	return tempIndex;
}

void printHashTable(myHashTable_t *table)
{
	int i = 0,j = 0;
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

//Returns 0 if entry1 is to be preferred, 1 if entry2 is to be preferred
//And -1 if both are identical.

int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple)
{
	if (((MovieTuple_t *) entry1->ptr)->movieRating > movieTuple->movieRating)
	{
		return 0;
	}
	else if (((MovieTuple_t *) entry1->ptr)->movieRating < movieTuple->movieRating)
	{
		return 1;
	}
	else if (((MovieTuple_t *) entry1->ptr)->movieVotes >  movieTuple->movieVotes)
	{
		return 0;
	}
	else if (((MovieTuple_t *) entry1->ptr)->movieVotes <  movieTuple->movieVotes)
	{
		return 1;
	}
	else 
	{
		return -1;
	}
}

void sortYearReleases(short yearArray[64],short releasesInYear[64],int numYears)
{
	short swapElement;
	int i,j;
	for (i = 0; i < numYears; i++)
	{
		for (j = i; j < numYears; j++)
		{
			if (releasesInYear[i] < releasesInYear[j])
			{
				swapElement = releasesInYear[j];
				releasesInYear[j] = releasesInYear[i];
				releasesInYear[i] = swapElement;

				swapElement = yearArray[j];
				yearArray[j] = yearArray[i];
				yearArray[i] = swapElement;
			}
		}
	}
}

void actualWorkFunction(char **lineBuff,int start,int end)
{
	char **tempKeys;
	MovieTuple_t tempTuple;
	char **splitBuff;
	int yearIndex,index;
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	int i;
	
	tempKeys = malloc(sizeof(char *) * 2);
	tempKeys[0] = malloc(sizeof(char) * 8);
	tempKeys[1] = malloc(sizeof(char) * 64);

	splitBuff = (char **) malloc(sizeof(char *) * 8);
	for(i = 0; i < 8;i++)
	{
		splitBuff[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}

	for (i = start; i < end; i++)
	{
		strcpy(tempBuff,lineBuff[i]);
		if (splitLine(tempBuff,splitBuff,":") != 5)
		{
			writeLog(__func__,VERROR,systemLogLevel,"Badly formed movie record: %s",tempBuff);
		}
		else
		{
			tempTuple.line = lineBuff[i];
			strcpy(tempTuple.movieName,splitBuff[0]);
			tempTuple.movieVotes = (unsigned int) atoi(splitBuff[1]);
			tempTuple.movieRating = (unsigned short) atoi(splitBuff[2]);
			tempTuple.movieYear = (unsigned short) atoi(splitBuff[3]);
			strcpy(tempTuple.movieCountry,splitBuff[4]);


			sprintf(tempKeys[0],"%d",tempTuple.movieYear);

			strcpy(tempKeys[1],tempTuple.movieCountry);

			index = shortArrayContains(globalYearArray,tempTuple.movieYear,numYears);
			if (index == -1)
			{
				globalReleasesInYear[numYears] = 1;
				globalYearArray[numYears++] = tempTuple.movieYear;
			}
			else
			{
				globalReleasesInYear[index]++;
			}
			index = arrayContains(countryArray,tempTuple.movieCountry,numCountries);
			yearIndex = shortArrayContains(countryYearArray[index],tempTuple.movieYear,countryNumYears[index]);
			if (yearIndex == -1)
			{
				countryReleasesInYear[index][countryNumYears[index]] = 1;
				countryYearArray[index][countryNumYears[index]] = tempTuple.movieYear;
				countryNumYears[index]++;
			}
			else
			{
				countryReleasesInYear[index][yearIndex]++;
			}
			addToHashTable(&hashTable,tempKeys,2,&tempTuple,&collisions,&numCollisions);
		}
	}
}
