/*******************************************************
cache.h
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#ifndef CACHE_H
#define CACHE_H

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <assert.h>
#include "defines.h"
#include "IMemoryController.h"
#include "ICache.h"

using namespace std;


class Cache: public ICache
{
protected:
	static int processorID;

	ulong size, lineSize, assoc, sets, log2Sets, log2Blk, tagMask, numLines;
	ulong reads,readMisses,writes,writeMisses,writeBacks;

	IMemoryController *controller;
	coherenceProtocol currentProtocol;

	//******///
	//add coherence counters here///
	//******
	ulong stateChangeMatrix[INVALID][INVALID];
	ulong interventions,flushes,invalidations;

	vector<vector<cacheLine> > cache;
	ulong calcTag(ulong addr)     { return (addr >> (log2Blk) );}
	ulong calcIndex(ulong addr)  { return ((addr >> log2Blk) & tagMask);}
	ulong calcAddr4Tag(ulong tag)   { return (tag << (log2Blk));}

public:
	ulong currentCycle;

	Cache(int cache_size,int cache_assoc,int blk_size);
	//~Cache() { delete cache;}

	cacheLine          *findLineToReplace(ulong addr);
	cacheLine          *fillLine(ulong addr);
	cacheLine          *findLine(ulong addr);
	cacheLine          *getLRU(ulong);
	virtual int         setState(ulong addr,cacheState newState,bool isFlushNeeded = false);
	virtual cacheState  getState(ulong addr);
	virtual bool        hasLine(ulong addr);

	ulong getRM(){return readMisses;}
	ulong getWM(){return writeMisses;} 
	ulong getReads(){return reads;}
	ulong getWrites(){return writes;}
	ulong getWB(){return writeBacks;}
	void  setProtocol(coherenceProtocol protocol){currentProtocol = protocol;}

	void writeBack(ulong)   {writeBacks++;}
	virtual void Access(ulong address,uchar operation);
	void printStats();
	void updateLRU(cacheLine *);
	void setController(IMemoryController &memController);
	virtual void recordStateChange(cacheState oldState,cacheState newState);
	virtual void snoopBusTransaction(ulong addr,busTransaction transaction);

	//******///
	//add other functions to handle bus transactions///
	//******///

};

#endif
