#ifndef DLIST_H
#define DLIST_H

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<omp.h>

 
#define NUM_ELEMENT 500
#define SEED 0

/* IntListNode structure defines a node in the linked list */
typedef struct tagIntListNode{
  int data;
  struct tagIntListNode *next;
  struct tagIntListNode *prev;
} IntListNode;
typedef IntListNode *pIntListNode;


/* IntList is a list of integer values, ascendingly sorted, with the 
   smallest integer value at the head of the list */
typedef struct {
  pIntListNode  head;
} IntList;
typedef IntList* pIntList;

typedef struct {
  pIntListNode nodes[NUM_ELEMENT];
  int curPtr;
  int numElmt;
} ArrNode;
typedef ArrNode *pArrNode;



pIntListNode IntListNode_Create(int x);

void IntList_Init(pIntList list);
void IntList_Insert(pIntList pList, int x, pArrNode an);
void IntList_Delete(pIntList pList, int x);
void IntList_Print(pIntList list);

void ArrNode_Init(pArrNode an, long num);
pIntListNode ArrNode_getNode(pArrNode an);

#endif
