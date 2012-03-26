#include <stdio.h>
//#include "util.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#define BUCKET_SIZE 512
#define MICRO_BUFFER_SIZE 32
#define SMALL_BUFFER_SIZE 128
#define MEDIUM_BUFFER_SIZE 256
#define BUFFER_SIZE 1024

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256

typedef struct MovieTuple
{
	char movieName[SMALL_BUFFER_SIZE];
	unsigned int movieVotes;
	unsigned short movieRating;
	unsigned short movieYear;
	char movieCountry[MICRO_BUFFER_SIZE];
	char *line;
} MovieTuple_t;


enum VLOGLEVEL
{
	NONE = 0,
	VINFO = 1,
	VWARNING = 2,
	VERROR = 4,
	VSEVERE = 8,
	VDEBUG = 15
};

typedef struct myHashEntry
{
	char **keys;
	void *ptr;
	int numKeys;
}myHashEntry_t;

typedef struct collidedEntry
{
	MovieTuple_t *tuple;
	unsigned int bucketIndex;
} collidedEntry_t;

typedef struct myHashTable
{
	myHashEntry_t entries[BUCKET_SIZE];
}myHashTable_t;

#ifdef _WIN32
#define my_strtok strtok_s
#else
#define my_strtok strtok_r
#endif

int splitLine(const char *in, char **out,const char *delim)
{
	int i = 0;
	char *ptr, *saveptr;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	strcpy(tempBuff,in);

	ptr = my_strtok(tempBuff,delim,&saveptr);
	while (ptr != NULL)
	{
		out[i][0] = '\0';
		strcpy(out[i],ptr);
		i++;
		ptr = my_strtok(NULL,delim,&saveptr);

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

int shortArrayContains(short *array, short element, int arrayLen)
{
	int i;
	for (i = 0; i < arrayLen; i++)
	{
		if (array[i] == element)
		{
			return i;
		}
	}
	return -1;
}


int writeLog(const char *sourceFunction,enum VLOGLEVEL loglevel,int _system_log_level,char *fmt, ...)
{
	if (_system_log_level != (int) NONE)
	{
		char text[512];
		char tempBuff[512];
		struct tm *timeVal;
		time_t currTime;
		char timeBuff[64];
		va_list argp;


		currTime = time(NULL);
		timeVal = localtime(&currTime);
		strftime(timeBuff,64,"%Y%m%d %H:%M:%S|",timeVal);
		strcpy(text,timeBuff);


		//LogFormat - Date|LOGLEVEL|SourceFunction|LogString
		if (loglevel == VERROR)
		{
			strcat(text,"ERROR|");
		}
		else if (loglevel == VWARNING)
		{
			strcat(text,"WARNING|");
		}
		else if (loglevel == VSEVERE)
		{
			strcat(text,"SEVERE|");
		}
		else if (loglevel == VINFO)
		{
			strcat(text,"INFO|");
		}
		else if (loglevel == VDEBUG)
		{
			strcat(text,"DEBUG|");
		}
		strcat(text,sourceFunction);
		strcat(text,"|");

		va_start(argp,fmt);
		vsprintf(tempBuff,fmt,argp);
		va_end(argp);

		strcat(text,tempBuff);
		strcat(text,"\n");
		if ( (_system_log_level & (int) loglevel) == (int) loglevel)
		{
			#ifdef _WIN32
				printf(text);
			#else
				write(1,text,strlen(text));
			#endif
		}
	}
	return 0;
}


int addToHashTable(myHashTable_t *table, char **keys, int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions);
unsigned int getFromHashTable(myHashTable_t *table, char **keys, int numKeys);
unsigned int hashFunction(unsigned int seed,char *key);
void printHashTable(myHashTable_t *table);
void initHashTable(myHashTable_t *table);
int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple);
void sortYearReleases(short yearArray[64],short releasesInYear[64],int numYears);

void actualWorkFunction(char **lineBuff,int start,int end);


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

collidedEntry_t *collisions;
int numCollisions = 0;
int totalMovies = 0;
MovieTuple_t *allMovies;


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
				totalMovies++;
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
						//Nice! so I will split the line, and put the unsplit line back in the buffer!
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
	allMovies = malloc(sizeof(MovieTuple_t) * totalMovies);
	collisions = malloc(sizeof(MovieTuple_t) * totalMovies);

#ifdef DEBUG
	printf("\nTotal Movies - %d ",totalMovies);
#endif

	for (i = 0; i < numCountries;i++)
	{
		actualWorkFunction(dataBuff[i],0,countryNumMovies[i]);
	}
	
	//Free dataBuff. We dont need it anymore
	for (i = 0; i < MAX_COUNTRIES; i++)
	{
		for (j = 0; j < LINES_PER_COUNTRY; j++)
		{
			free(dataBuff[i][j]);
		}
		free(dataBuff[i]);
	}
	for (i = 0; i < 8; i++)
	{
		free(splitBuff[i]);
	}

	free(splitBuff);
	free(dataBuff);

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
						printf("\n%s",collisions[k].tuple->line);
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
				//For some reason same movie is being inserted.
				////Handled upstream also;
				if (strcmp(collisions[k].tuple->movieName,highestRatedMovie->movieName) != 0)
				{
					printf("\n%s",collisions[k].tuple->line);
				}
			}
		}
	}
	printf("\n\n");
	for (i = 0; i < MAX_COUNTRIES; i++)
	{
		free(countryArray[i]);
	}
	free(countryArray);
	free(tempKeys[0]);
	free(tempKeys[1]);
	free(tempKeys);
	/*for ( i = 0; i < totalMovies; i++)
	{
		free(allMovies[i].line);
	}*/
	free(allMovies);
	free(collisions);
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

		if (strcmp(((MovieTuple_t *)(table->entries[bucketIndex].ptr))->movieName,ptr->movieName) != 0)
		{
			if ( collisionBreaker(&table->entries[bucketIndex],ptr) == 1)
			{
				memcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t));
			}
			else if ( collisionBreaker(&table->entries[bucketIndex],ptr) == -1 && *numCollisions < 256 )
			{
				//memcpy(&collisions[(*numCollisions)].tuple,ptr,sizeof(MovieTuple_t));
				collisions[(*numCollisions)].tuple = ptr;
				collisions[(*numCollisions)].bucketIndex = bucketIndex;
				(*numCollisions)++;
			}
			return 0;
		}
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
	char **splitBuffer;
	int yearIndex,index;
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	int i;
	static int movieCounter = 0;

	tempKeys = malloc(sizeof(char *) * 2);
	tempKeys[0] = malloc(sizeof(char) * 8);
	tempKeys[1] = malloc(sizeof(char) * 64);

	splitBuffer = (char **) malloc(sizeof(char *) * 8);
	for(i = 0; i < 8;i++)
	{
		splitBuffer[i] = (char *) malloc ( MAX_WORD_LENGTH * sizeof(char));
	}

	for (i = start; i < end; i++)
	{
		strcpy(tempBuff,lineBuff[i]);
		if (splitLine(tempBuff,splitBuffer,":") != 5)
		{
			writeLog(__func__,VERROR,systemLogLevel,"Badly formed movie record: %s",tempBuff);
		}
		else
		{
			allMovies[movieCounter].line = strdup(lineBuff[i]);
			strcpy(allMovies[movieCounter].movieName,splitBuffer[0]);
			allMovies[movieCounter].movieVotes = (unsigned int) atoi(splitBuffer[1]);
			allMovies[movieCounter].movieRating = (unsigned short) atoi(splitBuffer[2]);
			allMovies[movieCounter].movieYear = (unsigned short) atoi(splitBuffer[3]);
			strcpy(allMovies[movieCounter].movieCountry,splitBuffer[4]);


			sprintf(tempKeys[0],"%d",allMovies[movieCounter].movieYear);

			strcpy(tempKeys[1],allMovies[movieCounter].movieCountry);

			index = shortArrayContains(globalYearArray,allMovies[movieCounter].movieYear,numYears);
			if (index == -1)
			{
				globalReleasesInYear[numYears] = 1;
				globalYearArray[numYears++] = allMovies[movieCounter].movieYear;
			}
			else
			{
				globalReleasesInYear[index]++;
			}
			index = arrayContains(countryArray,allMovies[movieCounter].movieCountry,numCountries);
			yearIndex = shortArrayContains(countryYearArray[index],allMovies[movieCounter].movieYear,countryNumYears[index]);
			if (yearIndex == -1)
			{
				countryReleasesInYear[index][countryNumYears[index]] = 1;
				countryYearArray[index][countryNumYears[index]] = allMovies[movieCounter].movieYear;
				countryNumYears[index]++;
			}
			else
			{
				countryReleasesInYear[index][yearIndex]++;
			}
			addToHashTable(&hashTable,tempKeys,2,&allMovies[movieCounter],collisions,&numCollisions);
		}
		movieCounter++;
	}
	for (i = 0; i < 8; i++)
	{
		free(splitBuffer[i]);
	}
	free(splitBuffer);
	free(tempKeys[0]);
	free(tempKeys[1]);
	free(tempKeys);
}

int __test_and_set(int *mutex)
{
	return atomicCAS(mutex,0,1);
}

int __test_test_and_set(int *mutex);

int mycuda_mutex_lock(int *lock)
{
	// Check the test_test_and_set value.
	// returns -1 if mutex value is LOCKED
	int status = __test_test_and_set(mutex);
	while (status == -1){
		// If locked, wait on mutex to become unlocked.
	}
	else if(status == 0)
	{
		// The test_test_and_set returns 0 when the lcok was previously
		// not set and now it is set. i.e. lock has been acquired.
		return 0;
	}
}

