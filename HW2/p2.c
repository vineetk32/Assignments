#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef struct list{
	char *item;
	struct list *next;
} Node;

Node *begin = NULL;
Node *end = NULL;

int searchThreads = 0;
int insertThreads = 0;
int deleteThreads = 0;

int systemLogLevel;

int addToList(char *item){
	int success = -1;
	char *buf = NULL;
	Node *newItem = NULL;

	newItem = (Node *) malloc(sizeof(Node));
	if(newItem == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	buf = (char *) malloc(strlen(item) + 1);
	if(buf == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	strcpy(buf,item);
	newItem->item = buf;
	newItem->next = NULL;

	if(begin == NULL || end == NULL){
		begin = newItem;
		end = newItem;
		return 0;
	}

	end->next = newItem;
	end = end->next;
	return 0;
}

void printList(){
	Node *temp = NULL;
	if(begin == NULL)
		return;

	temp = begin;

	printf("==============\n");
	while(temp!=NULL){
		printf("%s --> " , temp->item);
		temp = temp->next;
	}
	printf("END");
	printf("\n==============");
}

int removeFromList(char *item){
	Node *temp = NULL;
	Node *prev = NULL;
	if(end == NULL){
		return -1;
	}

	temp = begin;
	prev = NULL;

	while(temp != NULL){
		if(strcmp(temp->item,item)== 0){
			if(prev == NULL){ 
				begin = begin->next;
				free(temp->item);
				free(temp);
				return 0;
				}
			
			if(temp->next == NULL){ 
				end = prev;
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

int searchList(char *item){
	Node * temp;
	temp = begin;
	
	if(temp == NULL){
		return -1;
	}

	while(temp != NULL){
		if(strcmp(temp->item,item) == 0){
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


int adder(char *line)
{
	if (line[0] == 'A')
	{
		addToList(line+2);
		return 0;
	}
	return -1;
}
int retriever(char *line)
{
	return searchList(line+2);
}
int deleter(char *line)
{
	if (line[0] == 'D')
	{
		removeFromList(line+2);
		return 0;
	}
	return -1;
}

int main(int argc , char** argv)
{
	FILE *finput;
	char line[BUFFER_SIZE] = {'\0'};

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
					adder(line);
					break;
				case 'D':
					deleter(line);
					break;
				case 'R':
					retriever(line);
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
	printList();
	return 0;
}

