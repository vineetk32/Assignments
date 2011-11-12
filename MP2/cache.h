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

	short stateChangeMatrix[INVALID][INVALID];

	IMemoryController *controller;
	coherenceProtocol currentProtocol;

	//******///
	//add coherence counters here///
	//******

	vector<vector<cacheLine>> cache;
	ulong calcTag(ulong addr)     { return (addr >> (log2Blk) );}
	ulong calcIndex(ulong addr)  { return ((addr >> log2Blk) & tagMask);}
	ulong calcAddr4Tag(ulong tag)   { return (tag << (log2Blk));}

public:
	ulong currentCycle;

	Cache(int cache_size,int cache_assoc,int blk_size);
	//~Cache() { delete cache;}

	cacheLine   *findLineToReplace(ulong addr);
	cacheLine   *fillLine(ulong addr);
	cacheLine   *findLine(ulong addr);
	cacheLine   *getLRU(ulong);
	int         setState(ulong addr,cacheState newState,bool isFlushNeeded = false);
	cacheState  getState(ulong addr);
	bool        hasLine(ulong addr);

	ulong getRM(){return readMisses;}
	ulong getWM(){return writeMisses;} 
	ulong getReads(){return reads;}
	ulong getWrites(){return writes;}
	ulong getWB(){return writeBacks;}
	void  setProtocol(coherenceProtocol protocol){currentProtocol = protocol;}

	void writeBack(ulong)   {writeBacks++;}
	void Access(ulong address,uchar operation);
	void printStats();
	void updateLRU(cacheLine *);
	void setController(IMemoryController &memController);
	void recordStateChange(cacheState oldState,cacheState newState);
	void snoopBusTransaction(ulong addr,busTransaction transaction);

	//******///
	//add other functions to handle bus transactions///
	//******///

};

#endif
