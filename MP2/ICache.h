/*******************************************************
cache.h
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#ifndef ICACHE_H
#define ICACHE_H

class ICache
{
public:
	virtual void Access(ulong address,uchar operation);
	virtual int setState(ulong addr,cacheState newState,bool isFlushNeeded);

	virtual cacheState  getState(ulong addr);
	virtual bool        hasLine(ulong addr);
	virtual void        recordStateChange(cacheState oldState,cacheState newState);
	virtual void        snoopBusTransaction(ulong addr,busTransaction transaction);

};

#endif
