#include "Cache.h"

MemoryController::MemoryController(int num_processors,coherenceProtocol protocol)
{
	currentProtocol = protocol;
}

void MemoryController::setCacheArray(vector<Cache> &newArray)
{
	cacheArray = &newArray;
}


void MemoryController::broadcastStateChange(ulong addr,cacheState newState)
{
	vector<Cache>::iterator it;
	for(it=cacheArray->begin(); it < cacheArray->end(); it++)
	{
		it->setState(addr,newState);
	}
}

bool MemoryController::copiesExist(ulong addr,int processorID)
{
	for (int i = 0; i < cacheArray->size(); i++)
	{
		if ( i != processorID)
		{
			if (cacheArray->at(i).hasLine(addr) == true)
			{
				return true;
			}
		}
	}
	return false;
}

//void MemoryController::requestBusTransaction(ulong addr,busTransaction transaction,int processorID)
int MemoryController::requestBusTransaction(ulong addr,busTransaction transaction,int processorID)
{
	int count = 0;
	for (int i = 0; i < cacheArray->size(); i++)
	{
		if ( i != processorID)
		{
			if ( cacheArray->at(i).hasLine(addr) == true)
			{
				if (cacheArray->at(i).snoopBusTransaction(addr,transaction) == true)
				{
					count++;
				}
			}
		}
	}
	return count;
}
int MemoryController::getNumCopies(ulong addr,int processorID)
{
	int count = 0;
	for (int i = 0; i < cacheArray->size(); i++)
	{
		if ( i != processorID)
		{
			if (cacheArray->at(i).hasLine(addr) == true)
			{
				count++;
			}
		}
	}
	return count;
}


