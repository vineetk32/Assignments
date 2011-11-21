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
#include <map>
#include "defines.h"


using namespace std;

class MemoryController;

class Cache
{
protected:
	int processorID;

	ulong size, lineSize, assoc, sets, log2Sets, log2Blk, tagMask, numLines;
	ulong reads,readMisses,writes,writeMisses,writeBacks,transfers;

	MemoryController *controller;
	coherenceProtocol currentProtocol;

	//******///
	//add coherence counters here///
	//******
	ulong stateChangeMatrix[INVALID+1][INVALID+1];
	ulong interventions,flushes,invalidations;

	vector<vector<cacheLine> > cache;
	//cacheLine **cache;

	ulong calcTag(ulong addr)     { return (addr >> (log2Blk) );}
	ulong calcIndex(ulong addr)  { return ((addr >> log2Blk) & tagMask);}
	ulong calcAddr4Tag(ulong tag)   { return (tag << (log2Blk));}

public:
	ulong currentCycle;

	Cache(int cache_size,int cache_assoc,int blk_size);
	void init(coherenceProtocol protocol,int processorID);
	//~Cache() { delete cache;}

	cacheLine   *findLineToReplace(ulong addr);
	cacheLine   *fillLine(ulong addr,processorAction action = PRRD);
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
	int   getProcessorID(){ return processorID; }

	void writeBack(ulong)   {writeBacks++;}
	void Access(ulong address,uchar operation);
	void printStats();
	void updateLRU(cacheLine *);
	void setController(MemoryController &memController);
	void recordStateChange(cacheState oldState,cacheState newState);
	bool snoopBusTransaction(ulong addr,busTransaction transaction);

};

class MemoryController
{
protected:
	int numProcessors;
	coherenceProtocol currentProtocol;
	vector<Cache> *cacheArray;

public:
	void setCacheArray(vector<Cache> &newArray);

	MemoryController(int num_processors,coherenceProtocol protocol);
	cacheState getBlockState(ulong addr);
	void setBlockState(ulong addr,cacheState newState);
	void addBlock(ulong addr);
	void broadcastStateChange(ulong addr,cacheState newState);
	bool copiesExist(ulong addr,int processorID);
	int requestBusTransaction(ulong addr,busTransaction transaction,int processorID);
	int getNumCopies(ulong addr,int processorID);
};

#endif
