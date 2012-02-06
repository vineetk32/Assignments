#include "string.h"

#ifndef __UTIL_H
#define __UTIL_H

int splitLine(const char *in, char **out,const char *delim);
int arrayContains(const char **array, const char *element, int arrayLen);

#endif