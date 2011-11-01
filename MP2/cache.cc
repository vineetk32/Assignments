/*******************************************************
cache.cc
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#include "cache.h"


Cache::Cache(int s,int a,int b )
{
	ulong i, j;
	reads = readMisses = writes = 0; 
	writeMisses = writeBacks = currentCycle = 0;

	size       = (ulong)(s);
	lineSize   = (ulong)(b);
	assoc      = (ulong)(a);
	sets       = (ulong)((s/b)/a);
	numLines   = (ulong)(s/b);
	log2Sets   = (ulong)(log2(int(sets)));
	log2Blk    = (ulong)(log2(int(b)));
	
	//*******************//
	//initialize your counters here//
	//*******************//

	tagMask =0;
	for(i=0;i<log2Sets;i++)
	{
		tagMask <<= 1;
		tagMask |= 1;
	}

	/**create a two dimentional cache, sized as cache[sets][assoc]**/ 
	//cache = new cacheLine*[sets];
	for(i=0; i<sets; i++)
	{
		vector<cacheLine> tempCache;
		for(j=0; j<assoc; j++) 
		{
			cacheLine tempBlock;
			tempBlock.invalidate();
			tempCache.push_back(tempBlock);
		}
		cache.push_back(tempCache);
	}
}

/**you might add other parameters to Access()
since this function is an entry point 
to the memory hierarchy (i.e. caches)**/
void Cache::Access(ulong addr,uchar op)
{
	currentCycle++;/*per cache global counter to maintain LRU order 
				   among cache ways, updated on every cache access*/

	if(op == 'w') writes++;
	else          reads++;

	cacheLine * line = findLine(addr);
	if(line == NULL)/*miss*/
	{
		if(op == 'w') writeMisses++;
		else readMisses++;

		cacheLine *newline = fillLine(addr);
		if(op == 'w') newline->setFlags(DIRTY);

	}
	else
	{
		/**since it's a hit, update LRU and update dirty flag**/
		updateLRU(line);
		if(op == 'w') line->setFlags(DIRTY);
	}
}

/*look up line*/
cacheLine * Cache::findLine(ulong addr)
{
	ulong i, j, tag, pos;

	pos = assoc;
	tag = calcTag(addr);
	i   = calcIndex(addr);

	for(j=0; j<assoc; j++)
		if(cache[i][j].isValid())
			if(cache[i][j].getTag() == tag)
			{
				pos = j; break; 
			}
			if(pos == assoc)
				return NULL;
			else
				return &(cache[i][pos]); 
}

/*upgrade LRU line to be MRU line*/
void Cache::updateLRU(cacheLine *line)
{
	line->setSeq(currentCycle);  
}

/*return an invalid line as LRU, if any, otherwise return LRU line*/
cacheLine * Cache::getLRU(ulong addr)
{
	ulong i, j, victim, min;

	victim = assoc;
	min    = currentCycle;
	i      = calcIndex(addr);

	for(j=0;j<assoc;j++)
	{
		if(cache[i][j].isValid() == 0) return &(cache[i][j]);
	}
	for(j=0;j<assoc;j++)
	{
		if(cache[i][j].getSeq() <= min) { victim = j; min = cache[i][j].getSeq();}
	}
	assert(victim != assoc);

	return &(cache[i][victim]);
}

/*find a victim, move it to MRU position*/
cacheLine *Cache::findLineToReplace(ulong addr)
{
	cacheLine * victim = getLRU(addr);
	updateLRU(victim);

	return (victim);
}

/*allocate a new line*/
cacheLine *Cache::fillLine(ulong addr)
{ 
	ulong tag;

	cacheLine *victim = findLineToReplace(addr);
	assert(victim != 0);
	if(victim->getFlags() == DIRTY) writeBack(addr);

	tag = calcTag(addr);
	victim->setTag(tag);
	victim->setFlags(VALID);
	/**note that this cache line has been already 
	upgraded to MRU in the previous function (findLineToReplace)**/

	return victim;
}

void Cache::printStats()
{ 
	printf("\n01. number of reads:				%ld",reads);
	printf("\n02. number of read misses:			%ld",readMisses);
	printf("\n03. number of writes:				%ld",writes);
	printf("\n04. number of write misses:			%ld",writeMisses);
	printf("\n05. number of write backs:			%ld",writeBacks);
	/*printf("\n06. number of invalid to exclusive (INV->EXC):	57
	printf("\n07. number of invalid to shared (INV->SHD):	36
	printf("\n08. number of modified to shared (MOD->SHD):	0
	printf("\n09. number of exclusive to shared (EXC->SHD):	20
	printf("\n10. number of shared to modified (SHD->MOD):	15
	printf("\n11. number of invalid to modified (INV->MOD):	13
	printf("\n12. number of exclusive to modified (EXC->MOD):	3
	printf("\n13. number of owned to modified (OWN->MOD):	0
	printf("\n14. number of modified to owned (MOD->OWN):	0
	printf("\n15. number of cache to cache transfers:		44
	printf("\n16. number of interventions:			0
	printf("\n17. number of invalidations:			21
	printf("\n18. number of flushes:				17*/
}
