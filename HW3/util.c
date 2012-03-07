#include "util.h"

int splitLine(const char *in, char **out,const char *delim)
{
	int i = 0;
	char *ptr, *saveptr;
	char tempBuff[BUFFER_SIZE] = {'\0'};
	strcpy(tempBuff,in);

	ptr = my_strtok(tempBuff,delim,&saveptr);
	while (ptr != NULL)
	{
		out[i][0] = '\0';
		strcpy(out[i],ptr);
		i++;
		ptr = my_strtok(NULL,delim,&saveptr);

	}
	return i;

}

int arrayContains(const char **array, const char *element, int arrayLen)
{
	int i;
	for (i = 0; i < arrayLen; i++)
	{
		if (strcmp(array[i],element) == 0)
		{
			return i;
		}
	}
	return -1;
}


int writeLog(const char *sourceFunction,enum VLOGLEVEL loglevel,int _system_log_level,char *fmt, ...)
{
	if (_system_log_level != (int) NONE)
	{
		char text[512];
		char tempBuff[512];
		struct tm *timeVal;
		time_t currTime;
		char timeBuff[64];
		va_list argp;
		

		currTime = time(NULL);
		timeVal = localtime(&currTime);
		strftime(timeBuff,64,"%Y%m%d %H:%M:%S|",timeVal);
		strcpy(text,timeBuff);

	
		//LogFormat - Date|LOGLEVEL|SourceFunction|LogString
		if (loglevel == VERROR)
		{
			strcat(text,"ERROR|");
		}
		else if (loglevel == VWARNING)
		{
			strcat(text,"WARNING|");
		}
		else if (loglevel == VSEVERE)
		{
			strcat(text,"SEVERE|");
		}
		else if (loglevel == VINFO)
		{
			strcat(text,"INFO|");
		}
		else if (loglevel == VDEBUG)
		{
			strcat(text,"DEBUG|");
		}
		strcat(text,sourceFunction);
		strcat(text,"|");
		
		va_start(argp,fmt);
		vsprintf(tempBuff,fmt,argp);
		va_end(argp);

		strcat(text,tempBuff);
		strcat(text,"\n");
		if ( (_system_log_level & (int) loglevel) == (int) loglevel)
		{
			#ifdef _WIN32
				printf(text);
			#else
				write(1,text,strlen(text));
			#endif
		}
	}
	return 0;
}

void initList(List_t *list)
{
	list->begin = NULL;
	list->end = NULL;
}

int addToList(List_t *list,void *item,size_t bytes){
	int success = -1;
	void *buf = NULL;
	Node_t *newItem = NULL;

	newItem = (Node_t *) malloc(sizeof(Node_t));
	if(newItem == NULL) {
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	buf = malloc(bytes);
	if(buf == NULL){
		fprintf(stderr,"addToList: Malloc failed");
		return -1;
	}

	memcpy(buf,item,bytes);
	//newItem->item = item;
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

char *ltrim(char *buffer,char *delims)
{
	int i = 0,j = 0;
	char *newBuff = buffer;
	while (1)
	{
		for (j = 0; j < strlen(delims); j++)
		{
			if (*buffer == delims[j])
			{
				newBuff++;
			}
			else if (*buffer == '\0')
			{
				return newBuff;
			}
		}
		newBuff++;
	}
	return newBuff;
}
char *rtrim(char *buffer,char *delims)
{
	//TODO
	return NULL;
}

