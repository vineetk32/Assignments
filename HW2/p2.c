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
	printf("\n==============\n");
}

int removeFromList(List_t *list,void *item,size_t bytes){
	Node_t *temp = NULL;
	Node_t *prev = NULL;
	if(list->end == NULL){
		return -1;
	}

	temp = list->begin;
	prev = NULL;

	//TODO: temp->item shouldnt have to be checked.
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
			else
			{
				prev->next = temp->next;
			}
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
		printf("Success %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n",operation,value,retriever_threads,adder_threads,deleter_threads);
	else
		printf("Fail %s [ %s ] Retrievers : %i Adders : %i Deleters : %i\n" , operation,value,retriever_threads,adder_threads,deleter_threads);

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

	success = addToList(_package->list,_package->line,strlen(_package->line));

#ifdef __DEBUG
	printf("Adder %u: Done\n",pthread_self());
#endif
	printThreadInfo("Adder",_package->line,success,pthread_self());
	
	pthread_mutex_lock(_package->adder_lock);
	adder_threads = 0;
	pthread_mutex_unlock(_package->adder_lock);
	
	pthread_cond_broadcast(_package->adder_cond);
	return NULL;
}

/* Retrievers walk through the linked list to find a particular item. Since they merely examine the list, there can be multiple retrievers accessing the list concurrently. */

void *retriever(void *package)
{
	int success = -1;
	retriever_thread_package_t *_package;
	_package = (retriever_thread_package_t *) package;

	pthread_mutex_lock(_package->deleter_lock);
	while (deleter_threads == 1)
	{
#ifdef __DEBUG
		printf("Retriever %u: waiting for the deleter.\n",pthread_self());
#endif
		pthread_cond_wait(_package->deleter_cond,_package->deleter_lock);
	}
	pthread_mutex_unlock(_package->deleter_lock);

	pthread_mutex_lock(_package->retriever_lock);
	retriever_threads++;
	pthread_mutex_unlock(_package->retriever_lock);

	success = searchList(_package->list,_package->line,strlen(_package->line));

#ifdef __DEBUG
	printf("Retriever %u: Done.\n",pthread_self());
#endif
	printThreadInfo("Retriever",_package->line,success,pthread_self());

	pthread_mutex_lock(_package->retriever_lock);
	retriever_threads--;
	pthread_mutex_unlock(_package->retriever_lock);

	pthread_cond_broadcast(_package->retriever_cond);
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
	while (adder_threads == 1)
	{
		pthread_cond_wait(_package->adder_cond,_package->adder_lock);
	}
	pthread_mutex_unlock(_package->adder_lock);

	pthread_mutex_lock(_package->retriever_lock);
	while (retriever_threads > 0)
	{
#ifdef __DEBUG
		printf("Deleter %u: waiting for a retriever.\n",pthread_self());
#endif
		pthread_cond_wait(_package->retriever_cond,_package->retriever_lock);
	}
	pthread_mutex_unlock(_package->retriever_lock);

	pthread_mutex_lock(_package->deleter_lock);
	while (deleter_threads == 1)
	{
		pthread_cond_wait(_package->deleter_cond,_package->deleter_lock);
	}
	deleter_threads = 1;
	pthread_mutex_unlock(_package->deleter_lock);

	success = removeFromList(_package->list,_package->line,strlen(_package->line));

#ifdef __DEBUG
	printf("Deleter %u: Done.\n",pthread_self());
#endif
	printThreadInfo("Deleter",_package->line,success,pthread_self());
	
	pthread_mutex_lock(_package->deleter_lock);
	deleter_threads = 0;
	pthread_mutex_unlock(_package->deleter_lock);

	pthread_cond_broadcast(_package->deleter_cond);
	return NULL;
}

void destroyThreadPackage(void *package)
{
	generic_thread_package_t *generic_package;
	retriever_thread_package_t *retriever_package;
	deleter_thread_package_t *deleter_package;

	generic_package = (generic_thread_package_t *) package;

	generic_package->list = NULL;
	free(generic_package->line);

	/*if (generic_package->type == DELETER)
	{
		deleter_package = (deleter_thread_package_t *) package;
		free(deleter_package->line);
		pthread_mutex_destroy(deleter_package->adder_lock);
		pthread_mutex_destroy(deleter_package->deleter_lock);
		pthread_mutex_destroy(deleter_package->retriever_lock);

		pthread_cond_destroy(deleter_package->adder_cond);
		pthread_cond_destroy(deleter_package->deleter_cond);
		pthread_cond_destroy(deleter_package->retriever_cond);
	}
	else if (generic_package->type == ADDER || generic_package->type == RETRIEVER)
	{
		retriever_package = (retriever_thread_package_t *) package;
		free(retriever_package->line);
		pthread_mutex_destroy(retriever_package->deleter_lock);
		pthread_mutex_destroy(retriever_package->retriever_lock);

		pthread_cond_destroy(retriever_package->deleter_cond);
		pthread_cond_destroy(retriever_package->retriever_cond);
	}
	else
	{
		fprintf(stderr,"Bad sizeof(package):%u. Expected %u or %u\n",sizeof(package),sizeof(retriever_thread_package_t),sizeof(deleter_thread_package_t));
		return;
	}*/
}

int main(int argc , char** argv)
{
	FILE *finput;
	char line[BUFFER_SIZE] = {'\0'};
	
	List_t data_list,thread_list,package_list;
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
	initList(&data_list);
	initList(&thread_list);
	initList(&package_list);

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
					adder_package->type = ADDER;
					adder_package->line = strdup(line+2);
					adder_package->list = &data_list;
					adder_package->adder_cond = &adder_cond;
					adder_package->adder_lock = &adder_lock;
					adder_package->deleter_cond = &deleter_cond;
					adder_package->deleter_lock = &deleter_lock;

					addToList(&thread_list,newThread,sizeof(pthread_t));
					addToList(&package_list,adder_package,sizeof(adder_thread_package_t));
#ifdef __DEBUG
					printf("Spawning adder thread. Adding %s to the list.\n",adder_package->line);
#endif
					pthread_create(newThread,NULL,adder,adder_package);
					break;
				case 'D':
					newThread = (pthread_t *) malloc(sizeof(pthread_t));
					deleter_package = (deleter_thread_package_t *) malloc(sizeof(deleter_thread_package_t));

					deleter_package->type = DELETER;
					deleter_package->line = strdup(line+2);
					deleter_package->list = &data_list;
					deleter_package->deleter_cond = &deleter_cond;
					deleter_package->deleter_lock = &deleter_lock;
					deleter_package->retriever_cond = &retriever_cond;
					deleter_package->retriever_lock = &retriever_lock;
					deleter_package->adder_cond = &adder_cond;
					deleter_package->adder_lock = &adder_lock;

					addToList(&thread_list,newThread,sizeof(pthread_t));
					addToList(&package_list,deleter_package,sizeof(deleter_thread_package_t));
#ifdef __DEBUG
					printf("Spawning deleter thread. Deleting %s from the list.\n",deleter_package->line);
#endif
					pthread_create(newThread,NULL,deleter,deleter_package);
#ifdef __DEBUG
					printf("Done deleting.\n",deleter_package->line);
#endif
					break;
				case 'R':
					newThread = (pthread_t *) malloc(sizeof(pthread_t));
					retriever_package = (retriever_thread_package_t *) malloc(sizeof(retriever_thread_package_t));

					retriever_package->type = RETRIEVER;
					retriever_package->line = strdup(line+2);
					retriever_package->list = &data_list;
					retriever_package->deleter_cond = &deleter_cond;
					retriever_package->deleter_lock = &deleter_lock;
					retriever_package->retriever_cond = &retriever_cond;
					retriever_package->retriever_lock = &retriever_lock;

					addToList(&thread_list,newThread,sizeof(pthread_t));
					addToList(&package_list,retriever_package,sizeof(retriever_thread_package_t));

#ifdef __DEBUG
					printf("Spawning retriever thread. Searching %s in the list.\n",retriever_package->line);
#endif
					pthread_create(newThread,NULL,retriever,retriever_package);
#ifdef __DEBUG
					printf("Done deleting.\n",retriever_package->line);
#endif
					break;
				case 'M':
					next_thread_node = thread_list.begin;
					next_package_node = package_list.begin;
#ifdef __DEBUG
					printf("Mark operation. Joining\n");
#endif
					while (next_thread_node != NULL)
					{
						curr_thread_node = next_thread_node;
						next_thread_node = next_thread_node->next;

						curr_package_node = next_package_node;
						next_package_node = next_package_node->next;
						pthread_join(*(pthread_t *)curr_thread_node->item,NULL);
						removeFromList(&thread_list,curr_thread_node,sizeof(curr_thread_node));
						//destroyThreadPackage(curr_package_node->item);
						removeFromList(&package_list,curr_package_node,sizeof(curr_package_node));
					}
					//TODO: Clean the queue in a proper way.
					//initList(&thread_list);
					//initList(&package_list);
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

	pthread_mutex_destroy(&adder_lock);
	pthread_mutex_destroy(&deleter_lock);
	pthread_mutex_destroy(&retriever_lock);

	pthread_cond_destroy(&adder_cond);
	pthread_cond_destroy(&deleter_cond);
	pthread_cond_destroy(&retriever_cond);

	printf("\nRead input. Final list - \n");
	printList(&data_list);
	return 0;
}


