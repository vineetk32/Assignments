#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef struct Node{
	void *item;
	struct Node *next;
} Node_t;

typedef struct List{
	Node_t *begin;
	Node_t *end;
} List_t;

int searchThreads = 0;
int insertThreads = 0;
int deleteThreads = 0;

int addToList(List_t *list,void *item,size_t bytes){
	int success = -1;
	void *buf = NULL;
	Node_t *newItem = NULL;

	newItem = (Node_t *) malloc(sizeof(Node_t));
	if(newItem == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	buf = malloc(bytes+1);
	if(buf == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	memcpy(buf,item,bytes+1);
	newItem->item = buf;
	newItem->next = NULL;

	if(list->begin == NULL || list->end == NULL){
		list->begin = newItem;
		list->end = newItem;
		return 0;
	}

	list->end->next = newItem;
	list->end = list->end->next;
	return 0;
}

void printList(List_t *list){
	Node_t *temp = NULL;
	if(list->begin == NULL)
		return;

	temp = list->begin;

	printf("==============\n");
	while(temp!=NULL){
		printf("%s --> " ,(char *)  temp->item);
		temp = temp->next;
	}
	printf("END");
	printf("\n==============");
}

int removeFromList(List_t *list,void *item,size_t bytes){
	Node_t *temp = NULL;
	Node_t *prev = NULL;
	if(list->end == NULL){
		return -1;
	}

	temp = list->begin;
	prev = NULL;

	while(temp != NULL){
		if(memcmp(temp->item,item,bytes)== 0){
			if(prev == NULL){ 
				list->begin = list->begin->next;
				free(temp->item);
				free(temp);
				return 0;
				}
			
			if(temp->next == NULL){ 
				list->end = prev;
			}

			prev->next = temp->next;
			free(temp->item);
			free(temp);
			return 0;
		}
		prev = temp;
		temp = temp->next;
	}
	return -1;
}

int searchList(List_t *list,char *item,size_t bytes){
	Node_t * temp;
	temp = list->begin;
	
	if(temp == NULL){
		return -1;
	}

	while(temp != NULL){
		if(memcmp(temp->item,item,bytes) == 0){
			return 0;
		}
		temp = temp->next;
	}
	return -1;
}

void printThreadInfo(char* operation, char* value, int success, pthread_t tid){
	if(success == 0)
		printf("[%08x]    Success %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n" ,tid, operation,value,searchThreads,insertThreads,deleteThreads);
	else	
		printf("[%08x]    Fail %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n" , tid , operation,value,searchThreads,insertThreads,deleteThreads);

}


int adder(List_t *list,char *line)
{
	if (line[0] == 'A')
	{
		addToList(list,line+2,strlen(line+2));
		return 0;
	}
	return -1;
}

int retriever(List_t *list,char *line)
{
	return searchList(list,line+2,strlen(line+2));
}
int deleter(List_t *list,char *line)
{
	if (line[0] == 'D')
	{
		removeFromList(list,line+2,strlen(line+2));
		return 0;
	}
	return -1;
}

int main(int argc , char** argv)
{
	FILE *finput;
	char line[BUFFER_SIZE] = {'\0'};
	List_t dataList;
	dataList.begin  = NULL;
	dataList.end = NULL;

	if (argc < 2)
	{
		fprintf(stderr,"\nInvalid Usage! Usage - %s <path-to-input-file>\n",argv[0]);
		return -1;
	}
	finput = fopen(argv[1],"r");
	if (finput == NULL)
	{
		fprintf(stderr,"\nFailed to open %s\n",argv[1]);
		return -1;
	}
	else
	{
		while (fgets(line,BUFFER_SIZE,finput) != NULL)
		{
			if (strlen(line) > 1)
			{
				line[strlen(line) - 1] = '\0';
				switch(line[0])
				{
				case 'A':
					adder(&dataList,line);
					break;
				case 'D':
					deleter(&dataList,line);
					break;
				case 'R':
					retriever(&dataList,line);
					break;
				case 'M':
					//Ignore for now.
					break;
				default:
					fprintf(stderr,"Invalid command:%c\n",line[0]);
					break;
				}
			}
			line[0] = '\0';
		}
		fclose(finput);
	}
	printf("\nRead input. Final list - \n");
	printList(&dataList);
	return 0;
}

