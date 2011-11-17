#ifndef IMEMCNTRLR_H
#define IMEMCNTRLR_H

#include "defines.h"

class IMemoryController
{
public:
	virtual cacheState getBlockState(ulong addr) = 0;
	virtual void setBlockState(ulong addr,cacheState newState);
	virtual void addBlock(ulong addr);
	virtual bool copiesExist(ulong addr,int processorID) ;

	virtual void broadcastStateChange(ulong addr,cacheState newState);
	virtual void requestBusTransaction(ulong addr,busTransaction transaction,int processorID);
	
};

#endif
