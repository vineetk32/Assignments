#include <stdio.h>
#include "util.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "myatomic.h"

#define MAX_CORPUS_WORDS 100
#define MAX_CORPUS_WORD_SIZE 100
#define BUCKET_SIZE 256

typedef struct myHashEntry
{
	char key[MICRO_BUFFER_SIZE];
	int value;
}myHashEntry_t;

typedef struct collidedEntry
{
	//MovieTuple_t *tuple;
	myHashEntry_t entry;
	unsigned int bucketIndex;
} collidedEntry_t;

typedef struct threadPackage
{
	int start,end;
	char **lineBuff;
	pthread_mutex_t *mutex;
} threadPackage_t;

typedef struct myHashTable
{
	myHashEntry_t entries[BUCKET_SIZE];
}myHashTable_t;


int __test_test_and_set(int *mutex);

void actualWorkFunction(char *dataBuff,int start,int end,myHashTable_t *table, collidedEntry_t *collisions, int *numCollisions,char **corpusWords,int numWords,myHashTable_t *fileHash,collidedEntry_t *fileCollisions, int *numFileCollisions);
unsigned int getFromHashTable(myHashTable_t *table, char *key);
unsigned long hashFunction(char *str);
void initHashTable(myHashTable_t *table);
void printHashTable(myHashTable_t *table);

