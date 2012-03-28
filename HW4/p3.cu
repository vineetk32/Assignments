#include <stdio.h>
//#include "util.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>


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

#define MUTEX_LOCKED 1
#define MUTEX_UNLOCKED 0

#ifdef _WIN32
#define my_strtok strtok_s
#else
#define my_strtok strtok_r
#endif

//#define DEBUG

#ifdef _WIN32
#define __func__ __FUNCTION__
#endif

//Modified from Dr. Xiaosong Ma's sample CUDA code provided in CSC548
#define CUDA_CALL(cmd) do { \
	if((err = cmd) != cudaSuccess) { \
		printf("(%d) Cuda Error: %s\n", __LINE__, cudaGetErrorString(err) ); \
		err = cudaSuccess; \
	} \
} while(0)


#define MAX_COUNTRIES 8
#define CHARS_PER_LINE 128
#define LINES_PER_COUNTRY 50000

#define THREADS_PER_BLOCK 2
#define BLOCKS_PER_GRID 4


int systemLogLevel;

int totalMovies = 0, numCollisions = 0;
int numCountries = 0,numYears = 0;
MovieTuple_t *allMovies;

MovieTuple_t *cudaAllMovies;
myHashTable_t *cudaHashTable;

collidedEntry_t *collisions;
collidedEntry_t *cudaCollisions;

char **countryArray;
char **cudaCountryArray;

short countryYearArray[MAX_COUNTRIES][64];
short **cudaCountryYearArray;

short countryReleasesInYear[MAX_COUNTRIES][64];
short **cudaCountryReleasesInYear;

int countryNumYears[MAX_COUNTRIES];
int *cudaCountryNumYears;

short globalYearArray[64];
short *cudaGlobalYearArray;

short globalReleasesInYear[64];
short *cudaGlobalReleasesInYear;


int __test_and_set(int *mutex)
{
//	return atomicCAS(mutex,0,1);
}


int addToHashTable(myHashTable_t *table, char **keys, int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions);
unsigned int getFromHashTable(myHashTable_t *table, char **keys, int numKeys);
unsigned int hashFunction(unsigned int seed,char *key);
void printHashTable(myHashTable_t *table);
void initHashTable(myHashTable_t *table);
int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple);
void sortYearReleases(short yearArray[64],short releasesInYear[64],int numYears);
void actualWorkFunction(char **lineBuff,int start,int end);
//void cudaSim(int blockIdxx,int blockDimx, int threadIDx,int totalLines);


int __test_test_and_set(int *mutex)
{
	// Test and set call. 100 tries.
	int counter = 100;
	while (counter > 0)
	{
		while (*mutex == MUTEX_LOCKED);
		if (__test_and_set(mutex)  == MUTEX_UNLOCKED)
		{
			return 0;
		}
		counter--;
	}
	return -1;
}

int mycuda_mutex_lock(int *lock)
{
	// Check the test_test_and_set value.
	// returns -1 if mutex valu* is LOCKED
	int status = __test_test_and_set(lock);
	while (status == -1){
		// If locked, wait on mutex to become unlocked.
	}
	if(status == 0)
	{
		// The test_test_and_set returns 0 when the lcok was previously
		// not set and now it is set. i.e. lock has been acquired.
		return 0;
	}
}

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

__device__ int shortArrayContains(short *array, short element, int arrayLen)
//int shortArrayContains(short *array, short element, int arrayLen)
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

//Taken from Salil Kanitkars HW1 solution.

/* CUDA device local func for string copy. */
__device__ void cudastrcpy(char *t, char *s)
//void cudastrcpy(char *t, char *s)
{
	while ( *s != '\0' ) {
		*t++ = *s++;
	}
	*t = '\0';
}

/* CUDA device local func for getting string length. */
__device__ int cudastrlen(char *src)
//int cudastrlen(char *src)
{
	int len=0;
	while ( *src++ != '\0' )
		len++;
	return (len);
}

/* CUDA device func for comparing strings. */
__device__ int cudastrcmp(char *s, char *d)
//int cudastrcmp(char *s, char *d)
{
	int len = cudastrlen(s), tmplen = cudastrlen(d);
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

__device__ int cudaArrayContains(const char **array, const char *element, int arrayLen)
//int cudaArrayContains(const char **array, const char *element, int arrayLen)
{
	int i;
	for (i = 0; i < arrayLen; i++)
	{
		if (cudastrcmp(array[i],element) == 0)
		{
			return i;
		}
	}
	return -1;
}


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
	int yearIndex;
	int highestBucketIndex = 0;
	int threadShare = 0;

	int countryNumMovies[MAX_COUNTRIES];
	myHashTable_t hashTable;

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

	//Allocate and copy all CUDA variables.

	//MovieTuple_t *cudaAllMovies;
	//myHashTable_t *cudaHashTable;
	//collidedEntry_t *cudaCollisions;
	//char **cudaCountryArray;
	//short **cudaCountryYearArray;
	//short **cudaCountryReleasesInYear;
	//int *cudaCountryNumYears;
	//short *cudaGlobalYearArray;
	//short *cudaGlobalReleasesInYear;

	//cudaAllMovies = (MovieTuple_t *) malloc(sizeof(MovieTuple_t) * totalMovies);
	CUDA_CALL(cudaMalloc((void **)&cudaAllMovies, totalMovies * sizeof(MovieTuple_t)));

	//cudaHashTable = (myHashTable_t *) malloc(sizeof(myHashTable_t));
	CUDA_CALL(cudaMalloc((void **)&cudaHashTable, sizeof(myHashTable_t)));

	//cudaCollisions = (collidedEntry_t *) malloc(sizeof(collidedEntry_t) * totalMovies);
	CUDA_CALL(cudaMalloc((void **)&cudaCollisions, sizeof(collidedEntry_t) * totalMovies));

	//cudaCountryArray = (char **) malloc(sizeof(char *) * numCountries);
	CUDA_CALL(cudaMalloc((void **)&cudaCountryArray, sizeof(char **) * numCountries));
	for ( i = 0 ; i < numCountries; i++)
	{
		//cudaCountryArray[i] = (char *) malloc(sizeof(char) * 32);
		CUDA_CALL(cudaMalloc((void **)&cudaCountryArray[i], sizeof(char) * 32));
	}

	//cudaCountryYearArray = (short **) malloc(sizeof(short *) * numCountries);
	CUDA_CALL(cudaMalloc((void **)&cudaCountryYearArray, sizeof(short *) * numCountries));
	for ( i = 0 ; i < numCountries; i++)
	{
		//cudaCountryYearArray[i] = (short *) malloc(sizeof(short) * numYears);
		CUDA_CALL(cudaMalloc((void **)&cudaCountryYearArray[i], sizeof(short) * numYears));
	}

	//cudaCountryReleasesInYear = (short **) malloc(sizeof(short *) * numCountries);
	CUDA_CALL(cudaMalloc((void **)&cudaCountryReleasesInYear, sizeof(short *) * numCountries));
	for ( i = 0 ; i < numCountries; i++)
	{
		//cudaCountryReleasesInYear[i] = (short *) malloc(sizeof(short) * numYears);
		CUDA_CALL(cudaMalloc((void **)&cudaCountryReleasesInYear[i], sizeof(short) * numYears));
	}

	//cudaCountryNumYears = (int *) malloc(sizeof(int) * numCountries);
	CUDA_CALL(cudaMalloc((void **)&cudaCountryYearArray, sizeof(int) * numYears));

	//cudaGlobalYearArray = (short *) malloc(sizeof(short) * numYears);
	CUDA_CALL(cudaMalloc((void **)&cudaGlobalYearArray, sizeof(short) * numYears));

	//cudaGlobalReleasesInYear = (short *) malloc(sizeof(short) * numYears);
	CUDA_CALL(cudaMalloc((void **)&cudaGlobalReleasesInYear, sizeof(short) * numYears));

	//Now copy all the variables
	//memcpy(cudaAllMovies,allMovies,sizeof(MovieTuple_t) * totalMovies);
	CUDA_CALL(cudaMemcpy(cudaAllMovies,allMovies, sizeof(MovieTuple_t) * totalMovies, cudaMemcpyHostToDevice));

	//memcpy(cudaHashTable,&hashTable,sizeof(myHashTable_t));
	CUDA_CALL(cudaMemcpy(cudaHashTable,hashTable, sizeof(myHashTable_t), cudaMemcpyHostToDevice));

	//memcpy(cudaCollisions,collisions,sizeof(collidedEntry_t) * totalMovies);
	CUDA_CALL(cudaMemcpy(cudaCollisions,collisions, sizeof(collidedEntry_t) * totalMovies, cudaMemcpyHostToDevice));

	for ( i = 0 ; i < numCountries; i++)
	{
		//memcpy(cudaCountryArray[i],countryArray[i],sizeof(char) * 32);
		CUDA_CALL(cudaMemcpy(countryArray[i],sizeof(char) * 32, cudaMemcpyHostToDevice));
		//memcpy(cudaCountryYearArray[i],countryYearArray[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(cudaCountryYearArray[i],countryYearArray[i],sizeof(char) * numYears, cudaMemcpyHostToDevice));
		//memcpy(cudaCountryArray[i],countryArray[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(cudaCountryArray[i],countryArray[i],sizeof(char) * numYears, cudaMemcpyHostToDevice));
		//memcpy(cudaCountryReleasesInYear[i],countryReleasesInYear[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(cudaCountryReleasesInYear[i],countryReleasesInYear[i],sizeof(char) * numCountries, cudaMemcpyHostToDevice));
	}

	//memcpy(cudaCountryNumYears,countryNumYears,sizeof(int) * numCountries);
	CUDA_CALL(cudaMemcpy(cudaCountryNumYears,countryNumYears,sizeof(int) * numCountries, cudaMemcpyHostToDevice));

	//memcpy(cudaGlobalYearArray,globalYearArray,sizeof(short) * numYears);
	CUDA_CALL(cudaMemcpy(cudaGlobalYearArray,globalYearArray,sizeof(short) * numYears, cudaMemcpyHostToDevice));

	//memcpy(cudaGlobalReleasesInYear,globalReleasesInYear,sizeof(short) * numYears);
	CUDA_CALL(cudaGlobalReleasesInYear,globalReleasesInYear,sizeof(short) * numYears, cudaMemcpyHostToDevice));
	threadShare = totalMovies / (BLOCKS_PER_GRID * THREADS_PER_BLOCK) + 1;
	
	//cudaSim(i,1,j,threadShare);
	threadFunc<<<1,1>>>(threadShare);

	//Copy everything back in place
	//memcpy(allMovies,cudaAllMovies,sizeof(MovieTuple_t) * totalMovies);
	CUDA_CALL(cudaMemcpy(allMovies,cudaAllMovies,sizeof(MovieTuple_t) * totalMovies, cudaMemcpyDeviceToHost));

	//memcpy(&hashTable,cudaHashTable,sizeof(myHashTable_t));
	CUDA_CALL(cudaMemcpy(&hashTable,cudaHashTable, sizeof(myHashTable_t), cudaMemcpyDeviceToHost));

	//memcpy(cudaCollisions,collisions,sizeof(collidedEntry_t) * totalMovies);
	CUDA_CALL(cudaMemcpy(collisions,cudaCollisions, sizeof(collidedEntry_t) * totalMovies, cudaMemcpyDeviceToHost));

	for ( i = 0 ; i < numCountries; i++)
	{
		//memcpy(countryArray[i],cudaCountryArray[i],sizeof(char) * 32);
		CUDA_CALL(cudaMemcpy(countryArray[i],cudaCountryArray[i],sizeof(char) * 32, cudaMemcpyDeviceToHost));
		//memcpy(countryYearArray[i],cudaCountryYearArray[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(countryYearArray[i],cudaCountryYearArray[i],sizeof(char) * numYears, cudaMemcpyDeviceToHost));
		//memcpy(countryArray[i],cudaCountryArray[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(countryArray[i],cudaCountryArray[i],sizeof(char) * numYears, cudaMemcpyDeviceToHost));
		//memcpy(countryReleasesInYear[i],cudaCountryReleasesInYear[i],sizeof(short) * numYears);
		CUDA_CALL(cudaMemcpy(countryReleasesInYear[i],cudaCountryReleasesInYear[i],sizeof(char) * numCountries, cudaMemcpyDeviceToHost));
	}

	//memcpy(countryNumYears,cudaCountryNumYears,sizeof(int) * numCountries);
	CUDA_CALL(cudaMemcpy(countryNumYears,cudaCountryNumYears,sizeof(int) * numCountries, cudaMemcpyDeviceToHost));

	//memcpy(globalYearArray,cudaGlobalYearArray,sizeof(short) * numYears);
	CUDA_CALL(cudaMemcpy(globalYearArray,cudaGlobalYearArray,sizeof(short) * numYears, cudaMemcpyDeviceToHost));

	//memcpy(globalReleasesInYear,cudaGlobalReleasesInYear,sizeof(short) * numYears);
	CUDA_CALL(globalReleasesInYear,cudaGlobalReleasesInYear,sizeof(short) * numYears, cudaMemcpyDeviceToHost));

	tempKeys = malloc(sizeof(char *) * 2);
	tempKeys[0] = malloc(sizeof(char) * 8);
	tempKeys[1] = malloc(sizeof(char) * 64);


#ifdef DEBUG
	printf("Years - ");
	for (i = 0; i < numYears; i++)
	{
		printf("\t%d",globalYearArray[i]);
	}
	printf("\n");
#endif

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
		printf("\n%d:%d:",globalYearArray[i],globalReleasesInYear[i]);
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
					highestBucketIndex = index;
				}
				else if ( ((MovieTuple_t *)tempEntry->ptr)->movieRating == highestRating)
				{
					if ( ((MovieTuple_t *)tempEntry->ptr)->movieVotes > highestRatedMovie->movieVotes)
					{
						highestRatedMovie = ((MovieTuple_t *)tempEntry->ptr);
						highestBucketIndex = index;
					}
				}
			}
			//Check collided Entries also

		}
		printf("\n%s",highestRatedMovie->line);
		for (k = 0 ; k < numCollisions;k++)
		{
			if (collisions[k].bucketIndex == highestBucketIndex)
			{
				//For some reason same movie is being inserted.
				////Handled upstream also;
				if (strcmp(collisions[k].tuple->movieName,highestRatedMovie->movieName) != 0)
				{
					printf("\n%s",collisions[k].tuple->line);
				}
			}
		}
		//printf("\n%s:%d:%d:%s",highestRatedMovie->movieName,highestRatedMovie->movieVotes,highestRatedMovie->movieRating,highestRatedMovie->movieCountry);
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

__device__ int addToHashTable(myHashTable_t *table, char keys[2][SMALL_BUFFER_SIZE], int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions)
//int addToHashTable(myHashTable_t *table, char keys[2][SMALL_BUFFER_SIZE], int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions)
{
	int i = 0;
	unsigned int bucketIndex = 0;
	for (i = 0; i < numKeys; i++)
	{
		bucketIndex += cudaHashFunction(bucketIndex,keys[i]);
	}
	bucketIndex = bucketIndex % BUCKET_SIZE;
	if (table->entries[bucketIndex].numKeys != 0)
	{
		//writeLog(__func__,VINFO,systemLogLevel,"Collision for keys %s%s and %s%s. (%d)",table->entries[bucketIndex].keys[0],table->entries[bucketIndex].keys[1],keys[0],keys[1],bucketIndex);

		if (cudastrcmp(((MovieTuple_t *)(table->entries[bucketIndex].ptr))->movieName,ptr->movieName) != 0)
		//if (strcmp(((MovieTuple_t *)(table->entries[bucketIndex].ptr))->movieName,ptr->movieName) != 0)
		{
			if ( collisionBreaker(&table->entries[bucketIndex],ptr) == 1)
			{
				CUDA_CALL(cudaMemcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t),cudaMemcpyDeviceToDevice));
				//memcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t));
				//writeLog(__func__,VINFO,systemLogLevel,"Replacing entry in HashTable.");
			}
			else if ( collisionBreaker(&(table->entries[bucketIndex]),ptr) == -1 && *numCollisions < totalMovies )
			{
				//memcpy(&collisions[(*numCollisions)].tuple,ptr,sizeof(MovieTuple_t));
				collisions[(*numCollisions)].tuple = ptr;
				collisions[(*numCollisions)].bucketIndex = bucketIndex;
				(*numCollisions)++;
				//writeLog(__func__,VINFO,systemLogLevel,"Adding entry to collision list at %d.",*numCollisions);
			}
			return 0;
		}
		return 0;
	}
	else
	{
		table->entries[bucketIndex].numKeys = numKeys;
		table->entries[bucketIndex].keys = (char **) malloc(sizeof(char *) * 2);
		table->entries[bucketIndex].keys[0] = (char *)  malloc (sizeof(char) * 8);
		table->entries[bucketIndex].keys[1] = (char *) malloc (sizeof(char) * 64);
		//strcpy(table->entries[bucketIndex].keys[0],keys[0]);
		cudastrcpy(table->entries[bucketIndex].keys[0],keys[0]);
		//strcpy(table->entries[bucketIndex].keys[1],keys[1]);
		cudastrcpy(table->entries[bucketIndex].keys[1],keys[1]);
		table->entries[bucketIndex].ptr = malloc(sizeof(MovieTuple_t));
		//memcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t));
		cudaMemcpy(table->entries[bucketIndex].ptr,ptr,sizeof(MovieTuple_t),cudaMemcpyDeviceToDevice);
	}
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

__device__ unsigned int cudaHashFunction(unsigned int seed,char *key)
{
	int tempIndex, i;
	tempIndex = seed;

	for (i = 0; i < cudaStrlen(key); i++)
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

//int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple)
__device__ int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple)
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
	char **splitBuffer;
	char tempBuff[MEDIUM_BUFFER_SIZE] = {'\0'};
	int i;
	static int movieCounter = 0;

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
		}
		movieCounter++;
	}
	for (i = 0; i < 8; i++)
	{
		free(splitBuffer[i]);
	}
	free(splitBuffer);
}

__global__ threadFunc(int totalLines)
{
	char tempKeys[2][SMALL_BUFFER_SIZE] = {'\0'};
	int index = 0, yearIndex = 0,i;
	int start,end;
	static int threadID = 0;

	threadID++;
	//threadID = blockIdxx * blockDimx + threadIDx;
	start = threadID * totalLines;
	if ( (start + totalLines) < totalMovies)
	{
		end = start + totalLines;
	}
	else
	{
		end = totalMovies;
	}
	printf("\nThread %d: Start: %d, End: %d",threadID,start,end);

	for (i = start; i < end; i++)
	{
		sprintf(tempKeys[0],"%d",cudaAllMovies[i].movieYear);
		strcpy(tempKeys[1],cudaAllMovies[i].movieCountry);

		index = shortArrayContains(cudaGlobalYearArray,cudaAllMovies[i].movieYear,numYears);
		if (index == -1)
		{
			cudaGlobalReleasesInYear[numYears] = 1;
			cudaGlobalYearArray[numYears++] = cudaAllMovies[i].movieYear;
		}
		else
		{
			cudaGlobalReleasesInYear[index]++;
		}
		index = arrayContains(cudaCountryArray,cudaAllMovies[i].movieCountry,numCountries);
		//index = cudaArrayContains(cudaCountryArray,cudaAllMovies[i].movieCountry,numCountries);
		yearIndex = shortArrayContains(cudaCountryYearArray[index],cudaAllMovies[i].movieYear,cudaCountryNumYears[index]);
		if (yearIndex == -1)
		{
			cudaCountryReleasesInYear[index][cudaCountryNumYears[index]] = 1;
			cudaCountryYearArray[index][cudaCountryNumYears[index]] = cudaAllMovies[i].movieYear;
			cudaCountryNumYears[index]++;
		}
		else
		{
			cudaCountryReleasesInYear[index][yearIndex]++;
		}
		//printf("\nDone with %d",i);
		addToHashTable(cudaHashTable,tempKeys,2,&allMovies[i],collisions,&numCollisions);
	}
}
