#include "string.h"

#ifndef __UTIL_H
#define __UTIL_H

#define MAX_WORDS 20
#define MAX_WORD_LENGTH 256
#define BUFFER_SIZE 1024


int splitLine(const char *in, char **out,const char *delim);
int arrayContains(const char **array, const char *element, int arrayLen);

#endif

