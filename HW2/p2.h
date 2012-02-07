#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

#ifndef __P2_H
#define __P2_H

typedef struct Node{
	void *item;
	struct Node *next;
} Node_t;

typedef struct List
{
	Node_t *begin;
	Node_t *end;
} List_t;

typedef struct retriever_thread_package
{
	List_t *list;
	char *line;
	pthread_mutex_t *deleterLock;
	pthread_cond_t *deleterCond;
} retriever_thread_package_t;

typedef struct adder_thread_package
{
	List_t *list;
	char *line;
	pthread_mutex_t *adderLock;
	pthread_cond_t *adderCond;
	pthread_mutex_t *deleterLock;
	pthread_cond_t *deleterCond;
} adder_thread_package_t;

typedef struct deleter_thread_package
{
	List_t *list;
	char *line;
	pthread_mutex_t *deleterLock;
	pthread_cond_t *deleterCond;
} deleter_thread_package_t;

void initList(List_t *list);
int addToList(List_t *list,void *item,size_t bytes);
void printList(List_t *list);
int removeFromList(List_t *list,void *item,size_t bytes);
int searchList(List_t *list,char *item,size_t bytes);
void printThreadInfo(char* operation, char* value, int success, pthread_t tid);

void *adder(void *package);
void *retriever(void *package);
void *deleter(void *package);

#endif