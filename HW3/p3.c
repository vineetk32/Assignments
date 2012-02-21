#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>


#define MAX_THREAD_NUM 5000
#define EXPIRE 3

typedef struct list{
char *item;
struct list *next;
}Node;

Node *begin = NULL;
Node *end = NULL;

int searchThreads = 0;
int insertThreads = 0;
int deleteThreads = 0;

bool addToList(char *item){
	bool success = false;
	
	Node *new = NULL;
	new = malloc(sizeof(Node));
	if(new == NULL){
		fprintf(stderr,"insertToList : Malloc failed");
		return false;
	}

	char * buf = malloc(strlen(item) + 1);
	if(buf == NULL){
		fprintf(stderr,"insertToList : Malloc failed");
		return false;
	}

	strcpy(buf,item);
	new->item = buf;
	new->next = NULL;

	if(begin == NULL || end == NULL){
		begin = new;
		end = new;
		return true;
	}

	end->next = new;
	end = end->next;
	return true;
}

void printList(){
	if(begin == NULL)
		return;

	Node *temp = begin;

	printf("==============\n");
	while(temp!=NULL){
		printf("%s\n" , temp->item);
		temp = temp->next;
	}
}

bool removeFromList(char *item){
	if(end == NULL){
		return false;
	}

	Node *temp = begin;
	Node *prev = NULL;

	while(temp != NULL){
		if(strcmp(temp->item,item)== 0){
			if(prev == NULL){ //first node
				begin = begin->next;
				free(temp->item);
				free(temp);
				return true;
				}
			
			if(temp->next == NULL){ //last node
				end = prev;
			}

			prev->next = temp->next;
			free(temp->item);
			free(temp);
			return true;
		}	
		prev = temp;	
		temp = temp->next;
	}
	return false;
}

bool searchList(char *item){
	Node * temp;
	temp = begin;
	
	if(temp == NULL){
		return false;
	}

	while(temp != NULL){
		if(strcmp(temp->item,item) == 0){
			return true;
		}
		temp = temp->next;
	}
	return false;
}

void printThreadInfo(char* operation, char* value, bool success, pthread_t tid){
	int len = strlen(value);
	value[len-1] = '\0'; //remove the endline char
	if(success)
		printf("[%08x]    Success %s [ %s ] Retrievers : %i Adders : %i Removers : %i\n" ,tid, operation,value,searchThreads,insertThreads,deleteThreads);
	else	
		printf("[%08x]    Fail %s [ %s ] Retrievers : %i Adders : %i Removers : %i\n" , tid , operation,value,searchThreads,insertThreads,deleteThreads);

}


void *searcher(void *args){
}

void* inserter(void *args){
}

void * deleter(void *args){
}


int main(int argc , char** argv){
	pthread_t th[MAX_THREAD_NUM];
	char* th_arg[MAX_THREAD_NUM];
	int counter=0;
	char c;
	int l;
	int j;
	int len;

	setbuf(stdout,NULL);

	 char filename[] = "p2_input.txt";
	 FILE *file = fopen ( filename, "r" );
	   if ( file != NULL )
	   {
	      char line [ 128 ]; /* or other suitable maximum line size */
	      while ( fgets ( line, sizeof line, file ) != NULL ) /* read a line */
      		{
		char *token; 
		token = strtok(line, " ");
		c = token[0];
                token = strtok(NULL, " ");
			 switch(c){
				case 'A' :		
					th_arg[counter] = malloc(strlen(token) * sizeof(char) + 1);
					strcpy(th_arg[counter],token);
					pthread_create(&(th[counter]), NULL, inserter, (void*)(th_arg[counter]));			
					counter++;
				break;
				
				case 'D' :
					th_arg[counter] = malloc(strlen(token) * sizeof(char) + 1);
					strcpy(th_arg[counter],token);
					pthread_create(&(th[counter]), NULL, deleter, (void*)(th_arg[counter]));			
					counter++;
				break;
				
				case 'M' :		
				for ( j = 0 ; j<counter ; j++ ){
						if(th[j] != NULL){
						pthread_join(th[j], NULL);
						th[j] = NULL;
						free(th_arg[j]);
					}
					}
				counter=0;
				break;
				
				case 'R' : 
					th_arg[counter] = malloc(strlen(token) * sizeof(char) + 1);
					strcpy(th_arg[counter],token);
					pthread_create(&(th[counter]), NULL, searcher, (void*)(th_arg[counter]));			
					counter++;
				break;
				
				default  :
					printf("Input error : %c", c);
			} 
		}
		printList();
		fclose ( file );
	   }	
	   else
	   {
	      perror ( filename ); /* why didn't the file open? */
	   }
	   return 0;
}
