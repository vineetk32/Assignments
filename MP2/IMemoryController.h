#ifndef IMEMCNTRLR_H
#define IMEMCNTRLR_H

#include "defines.h"

class IMemoryController
{
public:
	virtual cacheState getBlockState(unsigned long addr) = 0;
	virtual void setBlockState(unsigned long addr,cacheState newState);
	virtual void addBlock(unsigned long addr);
	virtual bool copiesExist(ulong addr,int processorID) ;

	virtual void broadcastStateChange(ulong addr,cacheState newState);
	virtual void requestBusTransaction(ulong addr,busTransaction transaction,int processorID);
	
};

#endif
