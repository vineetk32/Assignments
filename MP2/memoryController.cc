#include "MemoryController.h"

MemoryController::MemoryController(int num_processors,coherenceProtocol protocol)
{
	currentProtocol = protocol;
}


void MemoryController::setBlockState(unsigned long addr,cacheState newState)
{
	mainMemory[addr].setState(newState);
}
cacheState MemoryController::getBlockState(unsigned long addr)
{
	return mainMemory[addr].getState();
}

void MemoryController::addBlock(unsigned long addr)
{
	MemoryBlock tempBlock;
	tempBlock.setState(EXCLUSIVE);
	mainMemory[addr] = tempBlock;
}

void MemoryController::setCacheArray(vector<ICache> &newArray)
{
	cacheArray = &newArray;
}


void MemoryController::broadcastStateChange(ulong addr,cacheState newState)
{
	vector<ICache>::iterator it;
	for(it=cacheArray->begin(); it < cacheArray->end(); it++)
	{
		it->changeState(addr,newState);
	}
}

bool MemoryController::copiesExist(ulong addr,int processorID)
{
	for (int i = 0; i < cacheArray->size(); i++)
	{
		if ( i == processorID) continue;
		if (cacheArray->at(i).hasLine(addr) == true)
		{
			return true;
		}
	}
	return false;
}

void MemoryController::requestBusTransaction(ulong addr,busTransaction transaction,int processorID)
{
	for (int i = 0; i < cacheArray->size(); i++)
	{
		if ( i == processorID) continue;
		if (cacheArray->at(i).hasLine(addr) == true)
		{
			cacheArray->at(i).snoopBusTransaction(addr,transaction);
		}
	}
}
