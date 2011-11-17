#ifndef MEMCNTRLR_H
#define MEMCNTRLR_H

#include <vector>
#include <map>
#include "IMemoryController.h"
#include "ICache.h"

using namespace std;

class MemoryController : public IMemoryController
{
protected:
	int numProcessors;
	map<unsigned long,MemoryBlock> mainMemory;
	coherenceProtocol currentProtocol;
	vector<ICache*> cacheArray;

public:
	void setCacheArray(vector<ICache *> &newArray);

	MemoryController(int num_processors,coherenceProtocol protocol);
	cacheState getBlockState(ulong addr);
	void setBlockState(ulong addr,cacheState newState);
	void addBlock(ulong addr);
	void broadcastStateChange(ulong addr,cacheState newState);
	bool copiesExist(ulong addr,int processorID);
	void requestBusTransaction(ulong addr,busTransaction transaction,int processorID);
};
#endif