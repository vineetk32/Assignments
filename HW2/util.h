#ifndef __UTIL_H
#define __UTIL_H

#include "string.h"

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

#ifdef _WIN32
#define my_strtok strtok_s
#else
#define my_strtok strtok_r
#endif


int splitLine(const char *in, char **out,const char *delim);
int arrayContains(const char **array, const char *element, int arrayLen);
int writeLog(const char *sourceFunction,enum VLOGLEVEL loglevel,int _systemLogLevel,char *logStr);

#endif

