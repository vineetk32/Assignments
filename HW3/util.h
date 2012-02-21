#ifndef __UTIL_H
#define __UTIL_H

#include "string.h"
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256
#define BUFFER_SIZE 1024

enum VLOGLEVEL
{
	NONE = 0,
	VINFO = 1,
	VWARNING = 2,
	VERROR = 4,
	VSEVERE = 8,
	VDEBUG = 16
};

typedef struct Node{
	void *item;
	struct Node *next;
} Node_t;

typedef struct List
{
	Node_t *begin;
	Node_t *end;
} List_t;

#ifdef _WIN32
#define my_strtok strtok_s
#else
#define my_strtok strtok_r
#endif


void initList(List_t *list);
int  addToList(List_t *list,void *item,size_t bytes);
void printList(List_t *list);
int  removeFromList(List_t *list,void *item,size_t bytes);
int  searchList(List_t *list,char *item,size_t bytes);

int splitLine(const char *in, char **out,const char *delim);
int arrayContains(const char **array, const char *element, int arrayLen);
int writeLog(const char *sourceFunction,enum VLOGLEVEL loglevel,int _systemLogLevel,char *fmt, ...);

#endif

