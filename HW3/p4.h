
#include <stdio.h>
#include "util.h"
#include <stdlib.h>

#ifndef _P4_H
#define _P4_H

#define BUCKET_SIZE 255

typedef struct MovieTuple
{
	char movieName[SMALL_BUFFER_SIZE];
	unsigned int movieVotes;
	unsigned short movieRating;
	unsigned short movieYear;
	char movieCountry[MICRO_BUFFER_SIZE];
} MovieTuple_t;

typedef struct myHashEntry
{
	char **keys;
	void *ptr;
	int numKeys;
}myHashEntry_t;

typedef struct myHashTable
{
	myHashEntry_t entries[BUCKET_SIZE];
}myHashTable_t;

int addToHashTable(myHashTable_t *table, myHashEntry_t *entry);
int getFromHashTable(myHashTable_t *table, myHashEntry_t *entry);
int hashFunction(int seed,char *key,int bucketSize);
void printHashTable(myHashTable_t *table);
#endif