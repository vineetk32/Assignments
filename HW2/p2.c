#include "p2.h"

//#define __DEBUG

int retriever_threads = 0;
int adder_threads = 0;
int deleter_threads = 0;

void initList(List_t *list)
{
	list->begin = NULL;
	list->end = NULL;
}

int addToList(List_t *list,void *item,size_t bytes){
	int success = -1;
	//void *buf = NULL;
	Node_t *newItem = NULL;

	newItem = (Node_t *) malloc(sizeof(Node_t));
	if(newItem == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	/*buf = malloc(bytes+1);
	if(buf == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}*/

	//memcpy(buf,item,bytes+1);
	newItem->item = item;
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
				//free(temp->item);
				free(temp);
				return 0;
				}
			
			if(temp->next == NULL){ 
				list->end = prev;
			}

			prev->next = temp->next;
			//free(temp->item);
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
		printf("[%u]    Success %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n" ,tid, operation,value,retriever_threads,adder_threads,deleter_threads);
	else
		printf("[%u]    Fail %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n" , tid , operation,value,retriever_threads,adder_threads,deleter_threads);

}

/*Adders add new items to the end of the list. Adding an element must be mutually exclusive meaning multiple adders cannot access the list concurrently. However, an adder thread can proceed in parallel with any number of retrievers.*/

void *adder(void *package)
{
	adder_thread_package_t *_package;
	int success = -1;
	_package = (adder_thread_package_t *) package;

	pthread_mutex_lock(_package->deleter_lock);
	while (deleter_threads == 1)
	{

#ifdef __DEBUG
		printf("Adder %u: waiting for the deleter.\n",pthread_self());
#endif
		pthread_cond_wait(_package->deleter_cond,_package->deleter_lock);
	}
	pthread_mutex_unlock(_package->deleter_lock);
	
	pthread_mutex_lock(_package->adder_lock);
	while (adder_threads == 1)
	{
#ifdef __DEBUG
		printf("Adder %u: waiting for an adder...\n",pthread_self());
#endif
		pthread_cond_wait(_package->adder_cond,_package->adder_lock);

	}
	adder_threads = 1;
	pthread_mutex_unlock(_package->adder_lock);

	//printThreadInfo("Adder",_package->line+2,addToList(_package->list,_package->line+2,strlen(_package->line+2)),pthread_self());
	success = addToList(_package->list,_package->line+2,strlen(_package->line+2));
	pthread_cond_signal(_package->adder_cond);
	adder_threads = 0;

#ifdef __DEBUG
	printf("Adder %u: Done\n",pthread_self());
#endif
	printThreadInfo("Adder",_package->line+2,success,pthread_self());
	return NULL;
}

/* Retrievers walk through the linked list to find a particular item. Since they merely examine the list, there can be multiple retrievers accessing the list concurrently. */

void *retriever(void *package)
{
	int success = -1;
	retriever_thread_package_t *_package;
	_package = (retriever_thread_package_t *) package;
	pthread_mutex_lock(_package->retriever_lock);
	retriever_threads++;
	pthread_mutex_unlock(_package->retriever_lock);

	pthread_mutex_lock(_package->deleter_lock);
	while (deleter_threads == 1)
	{
#ifdef __DEBUG
		printf("Retriever %u: waiting for the deleter.\n",pthread_self());
#endif
		pthread_cond_wait(_package->deleter_cond,_package->deleter_lock);
	}
	pthread_mutex_unlock(_package->deleter_lock);

	//printThreadInfo("Adder",_package->line+2,searchList(_package->list,_package->line+2,strlen(_package->line+2)),pthread_self());
	success = searchList(_package->list,_package->line+2,strlen(_package->line+2));
	pthread_cond_signal(_package->retriever_cond);
	//TODO: call printThreadInfo
#ifdef __DEBUG
	printf("Retriever %u: Done.\n",pthread_self());
#endif
	printThreadInfo("Retriever",_package->line+2,success,pthread_self());
	return NULL;
}

/* Finally, deleters remove items from anywhere in the list. At most one deleter can access the list at a time and removing an item must be mutually exclusive with retrievals and additions. */

void *deleter(void *package)
{
	int success = -1;
	deleter_thread_package_t *_package;
	_package = (deleter_thread_package_t *) package;
	
	//TODO: Not the right way. Fix this.
	pthread_mutex_lock(_package->adder_lock);
	if (adder_threads == 1)
	{
		pthread_cond_wait(_package->adder_cond,_package->adder_lock);
	}
	pthread_mutex_unlock(_package->adder_lock);

	pthread_mutex_lock(_package->retriever_lock);
	if (retriever_threads == 1)
	{
#ifdef __DEBUG
		printf("Deleter %u: waiting for a deleter.\n",pthread_self());
#endif
		pthread_cond_wait(_package->retriever_cond,_package->retriever_lock);
	}
	pthread_mutex_unlock(_package->retriever_lock);

	pthread_mutex_lock(_package->deleter_lock);
	if (deleter_threads == 1)
	{
		pthread_cond_wait(_package->deleter_cond,_package->deleter_lock);
	}
	deleter_threads = 1;
	pthread_mutex_unlock(_package->deleter_lock);

	//printThreadInfo("Adder",_package->line+2,removeFromList(_package->list,_package->line+2,strlen(_package->line+2)),pthread_self());
	success = removeFromList(_package->list,_package->line+2,strlen(_package->line+2));
	pthread_cond_signal(_package->deleter_cond);
	deleter_threads = 0;

#ifdef __DEBUG
	printf("Deleter %u: Done.\n",pthread_self());
#endif
	printThreadInfo("Deleter",_package->line+2,success,pthread_self());
	return NULL;
}

int main(int argc , char** argv)
{
	FILE *finput;
	char line[BUFFER_SIZE] = {'\0'};
	
	List_t dataList,threadList,packageList;
	pthread_t *newThread = NULL;
	retriever_thread_package_t *retriever_package = NULL;
	adder_thread_package_t *adder_package = NULL;
	deleter_thread_package_t *deleter_package = NULL;
	Node_t *curr_thread_node = NULL;
	Node_t *next_thread_node = NULL;
	Node_t *curr_package_node = NULL;
	Node_t *next_package_node = NULL;
	pthread_mutex_t adder_lock;
	pthread_cond_t adder_cond;
	pthread_mutex_t deleter_lock;
	pthread_cond_t deleter_cond;
	pthread_mutex_t retriever_lock;
	pthread_cond_t retriever_cond;

	//Init all the queues,locks and condition vars
	initList(&dataList);
	initList(&threadList);
	initList(&packageList);

	pthread_mutex_init(&adder_lock,NULL);
	pthread_mutex_init(&deleter_lock,NULL);
	pthread_mutex_init(&retriever_lock,NULL);
	pthread_cond_init(&adder_cond,NULL);
	pthread_cond_init(&deleter_cond,NULL);
	pthread_cond_init(&retriever_cond,NULL);

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
			retriever_package = NULL;
			adder_package = NULL;
			deleter_package = NULL;

			if (strlen(line) > 1)
			{
				line[strlen(line) - 1] = '\0';
				switch(line[0])
				{
				case 'A':
					newThread = (pthread_t *) malloc(sizeof(pthread_t));
					adder_package = (adder_thread_package_t *) malloc(sizeof(adder_thread_package_t));
					adder_package->line = strdup(line);
					adder_package->list = &dataList;
					adder_package->adder_cond = &adder_cond;
					adder_package->adder_lock = &adder_lock;
					adder_package->deleter_cond = &deleter_cond;
					adder_package->deleter_lock = &deleter_lock;

					addToList(&threadList,newThread,sizeof(newThread));
					addToList(&packageList,adder_package,sizeof(adder_package));
#ifdef __DEBUG
					printf("Spawning adder thread. Adding %s to the list.\n",line+2);
#endif
					pthread_create(newThread,NULL,adder,adder_package);
					break;
				case 'D':
					newThread = (pthread_t *) malloc(sizeof(pthread_t));
					deleter_package = (deleter_thread_package_t *) malloc(sizeof(deleter_thread_package_t));
					deleter_package->line = strdup(line);
					deleter_package->list = &dataList;
					deleter_package->deleter_cond = &deleter_cond;
					deleter_package->deleter_lock = &deleter_lock;
					deleter_package->retriever_cond = &retriever_cond;
					deleter_package->retriever_lock = &retriever_lock;
					deleter_package->adder_cond = &adder_cond;
					deleter_package->adder_lock = &adder_lock;

					addToList(&threadList,newThread,sizeof(newThread));
					addToList(&packageList,deleter_package,sizeof(deleter_package));
#ifdef __DEBUG
					printf("Spawning deleter thread. Deleting %s from the list.\n",line+2);
#endif
					pthread_create(newThread,NULL,deleter,deleter_package);
#ifdef __DEBUG
					printf("Done deleting.\n",line+2);
#endif
					break;
				case 'R':
					newThread = (pthread_t *) malloc(sizeof(pthread_t));
					retriever_package = (retriever_thread_package_t *) malloc(sizeof(retriever_thread_package_t));
					retriever_package->line = strdup(line);
					retriever_package->list = &dataList;
					retriever_package->deleter_cond = &deleter_cond;
					retriever_package->deleter_lock = &deleter_lock;
					retriever_package->retriever_cond = &retriever_cond;
					retriever_package->retriever_lock = &retriever_lock;

					addToList(&threadList,newThread,sizeof(newThread));
					addToList(&packageList,retriever_package,sizeof(retriever_package));

#ifdef __DEBUG
					printf("Spawning retriever thread. Searching %s in the list.\n",line+2);
#endif
					pthread_create(newThread,NULL,retriever,retriever_package);
#ifdef __DEBUG
					printf("Done deleting.\n",line+2);
#endif
					break;
				case 'M':
					next_thread_node = threadList.begin;
					next_package_node = packageList.begin;
#ifdef __DEBUG
					printf("Mark operation. Joining\n",line[2]);
#endif
					while (next_thread_node != NULL)
					{
						curr_thread_node = next_thread_node;
						next_thread_node = next_thread_node->next;

						curr_package_node = next_package_node;
						next_package_node = next_package_node->next;
/*#ifdef __DEBUG
						printf("Waiting for thread %u...", *(pthread_t *)curr_thread_node->item);
#endif*/
						pthread_join(*(pthread_t *)curr_thread_node->item,NULL);
/*#ifdef __DEBUG
						printf("done.\n");
#endif*/
						removeFromList(&threadList,curr_thread_node,sizeof(curr_thread_node));
						removeFromList(&packageList,curr_package_node,sizeof(curr_package_node));
						//TODO: destroy mutexes and conds
					}
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


