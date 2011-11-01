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

inline long log2(int x) { return long(  log(double(x)) / log (double(2))); }

using namespace std;

typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned int uint;

/****add new states, based on the protocol****/
enum{
	INVALID = 0,
	VALID,
	DIRTY
};

class cacheLine 
{
protected:
	ulong tag;
	ulong Flags;   // 0:invalid, 1:valid, 2:dirty 
	ulong seq; 

public:
	cacheLine()                { tag = 0; Flags = 0; }
	ulong getTag()             { return tag; }
	ulong getFlags()           { return Flags;}
	ulong getSeq()             { return seq; }
	void setSeq(ulong Seq)     { seq = Seq;}
	void setFlags(ulong flags) {  Flags = flags;}
	void setTag(ulong a)       { tag = a; }
	void invalidate()          { tag = 0; Flags = INVALID; }//useful function
	bool isValid()             { return ((Flags) != INVALID); }
};

class Cache
{
protected:
	ulong size, lineSize, assoc, sets, log2Sets, log2Blk, tagMask, numLines;
	ulong reads,readMisses,writes,writeMisses,writeBacks;

	//******///
	//add coherence counters here///
	//******///

	vector<vector<cacheLine>> cache;
	ulong calcTag(ulong addr)     { return (addr >> (log2Blk) );}
	ulong calcIndex(ulong addr)  { return ((addr >> log2Blk) & tagMask);}
	ulong calcAddr4Tag(ulong tag)   { return (tag << (log2Blk));}

public:
	ulong currentCycle;  

	Cache(int cache_size,int cache_assoc,int blk_size);
	//~Cache() { delete cache;}

	cacheLine *findLineToReplace(ulong addr);
	cacheLine *fillLine(ulong addr);
	cacheLine *findLine(ulong addr);
	cacheLine *getLRU(ulong);

	ulong getRM(){return readMisses;} ulong getWM(){return writeMisses;} 
	ulong getReads(){return reads;}ulong getWrites(){return writes;}
	ulong getWB(){return writeBacks;}

	void writeBack(ulong)   {writeBacks++;}
	void Access(ulong address,uchar operation);
	void printStats();
	void updateLRU(cacheLine *);

	//******///
	//add other functions to handle bus transactions///
	//******///

};

#endif
