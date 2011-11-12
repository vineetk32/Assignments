/*******************************************************
                          main.cc
                  Ahmad Samih & Yan Solihin
                           2009
                {aasamih,solihin}@ece.ncsu.edu
********************************************************/

#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <fstream>
using namespace std;

#include "Cache.h"
#include "MemoryController.h"

int main(int argc, char *argv[])
{
	ifstream fin;
	FILE * pFile;

	char tempBuff[128] = {'\0'};
	int cacheNum;
	char cacheOperation;
	int i;
	vector<Cache> cacheVector;

	if(argv[1] == NULL){
		 printf("input format: ");
		 printf("./smp_cache <cache_size> <assoc> <block_size> <num_processors> <protocol> <trace_file> \n");
		 exit(0);
	}

	/*****uncomment the next five lines*****/
	//TODO:add validation
	int cache_size = atoi(argv[1]);
	int cache_assoc= atoi(argv[2]);
	int blk_size   = atoi(argv[3]);
	int num_processors = atoi(argv[4]);/*1, 2, 4, 8*/
	int protocol   = atoi(argv[5]);	 /*0:MSI, 1:MESI, 2:MOESI*/
	char *fname =  (char *)malloc(20);
	fname = argv[6];

	
	for(int i = 0; i < num_processors;i++)
	{
		Cache cacheObj(cache_size,cache_assoc,blk_size);
		cacheVector.push_back(cacheObj);
	}

	//****************************************************//
	//**printf("===== Simulator configuration =====\n");**//
	//*******print out simulator configuration here*******//
	//****************************************************//

	printf("\n===== 506 SMP Simulator Configuration =====");
	printf("\nL1_SIZE:\t\t\t%d",cache_size);
	printf("\nL1_ASSOC:\t\t\t%d",cache_assoc);
	printf("\nL1_BLOCKSIZE:\t\t\t%d",blk_size);
	printf("\nNUMBER OF PROCESSORS:\t\t%d",num_processors);
	printf("\nCOHERENCE PROTOCOL:\t\t%d",protocol);
	printf("\nTRACE FILE:\t\t\t %s",fname);
 
	//*********************************************//
	//*****create an array of caches here**********//
	//*********************************************//


	pFile = fopen (fname,"r");
	if(pFile == 0)
	{
		printf("Trace file problem\n");
		exit(0);
	}
	///******************************************************************//
	//**read trace file,line by line,each(processor#,operation,address)**//
	//*****propagate each request down through memory hierarchy**********//
	//*****by calling cachesArray[processor#]->Access(...)***************//
	///******************************************************************//
	while(fscanf(pFile,"%d %c %s",&cacheNum,&cacheOperation,tempBuff) > 0)
	{
		//printf("\ncacheNum: %d,cacheOperation: %c,memAddress: %s",cacheNum,cacheOperation,tempBuff);
		cacheVector[cacheNum].Access(strtoul(tempBuff,NULL,16),cacheOperation);
	}

	fclose(pFile);

	//********************************//
	//print out all caches' statistics //
	//********************************//
	for( i = 0;i<num_processors;i++)
	{
		printf("\n===== Simulation results (Cache_%d)      =====",i);
		cacheVector[i].printStats();
	}
	return 0;
}
