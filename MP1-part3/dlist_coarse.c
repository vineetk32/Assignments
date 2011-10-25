/***************************************************************
*  a simple doubly linked list implementation
*     
****************************************************************/
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include"omp.h"
#include"time.h"
#include"dlist.h"

// #include<omp.h>

omp_lock_t l;
#define NUM_THREADS 1

struct timespec start_time, end_time;
double elapsed;

pIntListNode IntListNode_Create(int x)
{
	pIntListNode newNode = (pIntListNode) malloc(sizeof(IntListNode));
	newNode->data = x;
	newNode->next = NULL;
	newNode->prev = NULL;
	newNode->deleted=0;
	return newNode;
}



void IntList_Init(pIntList list) 
{
	list->head = NULL;
}



void IntList_Insert(pIntList pList, int x, pArrNode an) 
{
	pIntListNode prev, p , newNode;
	// assert(newNode!=NULL);

	//#pragma omp critical


	newNode = ArrNode_getNode(an);

	omp_set_lock(&l) ;
	if (pList->head == NULL) { /* list is empty, insert the first element */
		pList->head = newNode;
		omp_unset_lock(&l);
	}


	else { /* list is not empty, find the right place to insert element */
		omp_unset_lock(&l);
		p = pList->head;
		prev = NULL;
		while (p != NULL && p->data < newNode->data) {
			prev = p;
			p = p->next;
		}


		omp_set_lock(&l) ;


		if(prev==NULL && p!=NULL && (p->deleted||p->prev!=NULL))
		{
			p = pList->head;
			prev = NULL;
			while (p != NULL && p->data < newNode->data) {
				prev = p;
				p = p->next;
			}

		}

		else if (prev!=NULL && p!=NULL && (p->deleted||prev->deleted||prev->next!=p||p->prev!=prev))
		{
			p = pList->head;
			prev = NULL;
			while (p != NULL && p->data < newNode->data) {
				prev = p;
				p = p->next;
			}

		}



		else if (prev!=NULL && p==NULL && (prev->deleted||prev->next!=NULL))
		{
			p = pList->head;
			prev = NULL;
			while (p != NULL && p->data < newNode->data) {
				prev = p;
				p = p->next;
			}

		}

		if (p == NULL) { /* insert as the last element */
			prev->next = newNode;
			newNode->prev = prev;
			newNode->deleted=0;
		}
		else if (prev == NULL) { /* insert as the first element */
			pList->head = newNode;
			newNode->next = p;
			newNode->deleted=0;
			p->prev = newNode;
		}
		else { /* insert right between prev and p */
			prev->next = newNode;
			newNode->prev = prev;
			newNode->next = p;
			newNode->deleted=0;
			p->prev = newNode;
		}
		omp_unset_lock(&l);

	}
	//end critical
}


/* delete the first element that has a data value equal to x */
void IntList_Delete(pIntList pList, int x) 
{
	pIntListNode prev, p, next;


	if (pList->head == NULL) { /* list is empty, do nothing */
	}
	else { /* list is not empty, find the desired element */
		p = pList->head;
		while (p != NULL && p->data != x) 
			p = p->next;

		if (p == NULL) { /* element not found, do nothing */
		}
		else {
			omp_set_lock(&l);


			if(p->prev==NULL && p!=NULL && p->next!=NULL && (p->deleted||(p->next)->deleted||p->next->prev!=p))
			{   
				p = pList->head;
				while (p != NULL && p->data != x) 
					p = p->next;
			}

			if(p->prev==NULL && p->next==NULL && p!=NULL && (p->deleted))
			{
				p = pList->head;
				while (p != NULL && p->data != x) 
					p = p->next;
			}

			if(p->prev!=NULL && p!=NULL && p->next!=NULL && (p->deleted||(p->next)->deleted||p->prev->deleted||p->next->prev!=p||p->prev->next!=p))
			{
				p = pList->head;
				while (p != NULL && p->data != x) 
					p = p->next;

			}

			if(p->prev!=NULL && p!=NULL && p->next ==NULL && (p->deleted || p->prev->deleted || p->prev->next!=p ) )
			{
				p = pList->head;
				while (p != NULL && p->data != x) 
					p = p->next;

			}





			if (p == NULL)  /* element not found in this re-traversal so do nothing */
			{

			}

			else  if (p->prev == NULL) { /* delete the head element */
				pList->head = p->next;
				if (p->next != NULL)
					p->next->prev = NULL;
				p->deleted=1;

			}
			else { /* delete non-head element */
				p->prev->next = p->next;
				if (p->next != NULL) {
					p->next->prev = p->prev;
				}
				p->deleted=1;

			}
			omp_unset_lock(&l);
		}
	}


}


void IntList_Print(pIntList list) 
{
	pIntListNode p = list->head;
	int i;

	if (p != NULL) {
		printf("\n --- Content of List --- \n");
		i = 0;
		printf("list's first element is %d: %d\n", i, p->data);
	}

	//while (p != NULL) {
	//	printf("list element %d: %d\n", i, p->data);
	//	p = p->next;
//		i++;
	//}
}




void ArrNode_Init(pArrNode an, long num) 
{
	int i;

	assert(num <= NUM_ELEMENT);

	for (i=0; i<num; i++) {
		int val = (int) 10000 * ((double)rand() / ((double)(RAND_MAX)+(double)(1)) ) ;
		an->nodes[i] = (pIntListNode) IntListNode_Create(val);
	}
	an->curPtr = 0;
	an->numElmt = num;
}

void ArrNode_Print(pArrNode an, long num)
{
	int i;

	for (i=0; i<num; i++) 
		printf("ArrNode_Print: element %d points to %d\n", i, an->nodes[i]);
}


pIntListNode ArrNode_getNode(pArrNode an)
	// return the value of the current node, advance curPtr
{
	assert(an->curPtr < NUM_ELEMENT);
	pIntListNode p = an->nodes[an->curPtr];
	an->curPtr++;
	return p; 
}


int main() 
{
	int i;
	pIntList pList = (pIntList) malloc(sizeof(IntList));
	pArrNode an1 = (pArrNode) malloc(sizeof(ArrNode));
	pArrNode an2 = (pArrNode) malloc(sizeof(ArrNode));
	omp_init_lock(&l);

	IntList_Init(pList);

	srand(SEED);

	ArrNode_Init(an1, NUM_ELEMENT);
	ArrNode_Init(an2, NUM_ELEMENT);
	// ArrNode_Print(an);


#pragma omp parallel sections default(shared) private(i)
	{
		clock_gettime(CLOCK_MONOTONIC,&start_time);
#pragma omp section
		{
			for (i=0; i<NUM_ELEMENT; i++)
				IntList_Insert(pList, i, an1);
		}
#pragma omp section
		{
			for (i=0; i<NUM_ELEMENT; i++)
				IntList_Insert(pList, NUM_ELEMENT-i, an2);
		}
		#pragma omp section
		{
		for (i=0; i<NUM_ELEMENT; i++)
		IntList_Delete(pList, 79*i % NUM_ELEMENT);
		}
		#pragma omp section
		{
		for (i=0; i<NUM_ELEMENT; i++)
		IntList_Delete(pList, 7919 * i % NUM_ELEMENT);
		}
	}
	clock_gettime(CLOCK_MONOTONIC,&end_time);
	elapsed=end_time.tv_sec-start_time.tv_sec;
	IntList_Print(pList);

	omp_destroy_lock(&l);
	printf("\nTime with %d threads: %lf\n",elapsed,NUM_THREADS);
}


