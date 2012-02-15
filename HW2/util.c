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

