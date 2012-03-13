
#include <stdio.h>
#include "util.h"
#include <stdlib.h>

#ifndef _P4_H
#define _P4_H

#define BUCKET_SIZE 512

typedef struct MovieTuple
{
	char movieName[SMALL_BUFFER_SIZE];
	unsigned int movieVotes;
	unsigned short movieRating;
	unsigned short movieYear;
	char movieCountry[MICRO_BUFFER_SIZE];
	char line[BUFFER_SIZE];
} MovieTuple_t;

typedef struct myHashEntry
{
	char **keys;
	void *ptr;
	int numKeys;
}myHashEntry_t;

typedef struct collidedEntry
{
	MovieTuple_t tuple;
	unsigned int bucketIndex;
} collidedEntry_t;

typedef struct myHashTable
{
	myHashEntry_t entries[BUCKET_SIZE];
}myHashTable_t;

int addToHashTable(myHashTable_t *table, char **keys, int numKeys,MovieTuple_t *ptr,collidedEntry_t *collisions,int *numCollisions);
unsigned int getFromHashTable(myHashTable_t *table, char **keys, int numKeys);
unsigned int hashFunction(unsigned int seed,char *key);
void printHashTable(myHashTable_t *table);
void initHashTable(myHashEntry_t *table);
int collisionBreaker(myHashEntry_t *entry1,MovieTuple_t *movieTuple);
void sortYearReleases(short yearArray[64],short releasesInYear[64],int numYears);

#endif
