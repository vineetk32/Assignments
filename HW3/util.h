#ifndef __UTIL_H
#define __UTIL_H

#include "string.h"
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256

#define MICRO_BUFFER_SIZE 32
#define SMALL_BUFFER_SIZE 128
#define MEDIUM_BUFFER_SIZE 256
#define BUFFER_SIZE 1024

//To make VS2010 shutup
//#define _CRT_SECURE_NO_WARNINGS
//#define _CRT_SECURE_NO_DEPRECATE

enum VLOGLEVEL
{
	NONE = 0,
	VINFO = 1,
	VWARNING = 2,
	VERROR = 4,
	VSEVERE = 8,
	VDEBUG = 15
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
int  searchList(List_t *list,void *item,size_t bytes);


int splitLine(const char *in, char **out,const char *delim);
int arrayContains(const char **array, const char *element, int arrayLen);
int shortArrayContains(short *array, short element, int arrayLen);
int writeLog(const char *sourceFunction,enum VLOGLEVEL loglevel,int _systemLogLevel,char *fmt, ...);
char *ltrim(char *buffer,char *delims);
char *rtrim(char *buffer,char *delims);



#endif

