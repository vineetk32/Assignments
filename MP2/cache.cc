/*******************************************************
cache.cc
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#include "Cache.h"

int Cache::processorID = 0;

Cache::Cache(int s,int a,int b)
{
	ulong i, j;
	reads = readMisses = writes = 0; 
	writeMisses = writeBacks = currentCycle = 0;
	invalidations = interventions = flushes = 0;

	size            = (ulong)(s);
	lineSize        = (ulong)(b);
	assoc           = (ulong)(a);
	sets            = (ulong)((s/b)/a);
	numLines        = (ulong)(s/b);
	log2Sets        = (ulong)(log2(int(sets)));
	log2Blk         = (ulong)(log2(int(b)));
	
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

	//Make the stateChangeMatrix 0;
	//Enum as a max_number :-O
	for(i = 0;i <= INVALID; i++)
	{
		for (j = 0; j <= INVALID; j++)
		{
			stateChangeMatrix[i][j] = 0;
		}
	}
	processorID++;
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

	cacheLine *line = findLine(addr);
	controller->getBlockState(addr);
	if(line == NULL)/*miss*/
	{
		if(op == 'w') writeMisses++;
		else readMisses++;

		cacheLine *newline = fillLine(addr);

		if(op == 'w')
		{
			controller->requestBusTransaction(addr,BUSRDX,processorID);
			if (currentProtocol == MSI)
			{
				//newline->setState(MODIFIED);
				setState(addr,MODIFIED);
				
			}
			else 
			{
				if (controller->copiesExist(addr,processorID) == true)
				{
					//newline->setSeq(MODIFIED);
					//TODO:update cache to cache transfer counter.
					setState(addr,MODIFIED);
				}
				else
				{
					//newline->setState(EXCLUSIVE);
					setState(addr,EXCLUSIVE);
				}
			}
		}
	}
	else
	{
		updateLRU(line);
		//if(op == 'w') line->setState(DIRTY);
		if(op == 'w')
		{
			cacheState currState = getState(addr);
			if (currentProtocol == MSI)
			{
				if (currState == SHARED || currState == INVALID)
				{
					controller->requestBusTransaction(addr,BUSRDX,processorID);
					setState(addr,MODIFIED);
				}
				
			}
			else if (currentProtocol == MESI)
			{
				if (currState == INVALID)
				{
					controller->requestBusTransaction(addr,BUSRDX,processorID);
					setState(addr,MODIFIED);
				}
				else if (currState == SHARED)
				{
					controller->requestBusTransaction(addr,BUSUPGR,processorID);
					setState(addr,MODIFIED);
				}
				setState(addr,MODIFIED);
			}
			else if (currentProtocol == MOESI)
			{
				//TODO
			}
		}
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
	{
		if(cache[i][j].isValid())
		{
			if(cache[i][j].getTag() == tag)
			{
				pos = j; break;
			}
			if(pos == assoc)
				return NULL;
			else
				return &(cache[i][pos]);
		}
	}
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
	cacheState newState;

	cacheLine *victim = findLineToReplace(addr);
	assert(victim != 0);

	if(currentProtocol == MOESI)
	{
		if (victim->getState() == OWNER)
		{
			//TODO:Check
			writeBack(addr);
			recordStateChange(OWNER,INVALID);
		}
	}
	else if (victim->getState() == MODIFIED)
	{
		//TODO:Check
		writeBack(addr);
		recordStateChange(MODIFIED,INVALID);
	}

	tag = calcTag(addr);
	victim->setTag(tag);
	if (controller->copiesExist(addr,processorID) == false)
	{
		if (currentProtocol == MSI)
		{
			newState = SHARED;
			recordStateChange(INVALID,SHARED);
		}
		else
		{
			newState = EXCLUSIVE;
			recordStateChange(INVALID,EXCLUSIVE);
		}
	}
	else
	{
		switch(controller->getBlockState(addr))
		{
		case MODIFIED:
		case OWNER:
		case EXCLUSIVE:
		case SHARED:
		case INVALID:
		default:
			cout<<"\nWhat?\n";
			exit(0);
		}
	}
	victim->setState(newState);
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
	printf("\n06. number of invalid to exclusive (INV->EXC):	%ld",stateChangeMatrix[INVALID][SHARED]);
	printf("\n07. number of invalid to shared (INV->SHD):	%ld",stateChangeMatrix[INVALID][SHARED]);
	printf("\n08. number of modified to shared (MOD->SHD):	%ld",stateChangeMatrix[MODIFIED][SHARED]);
	printf("\n09. number of exclusive to shared (EXC->SHD):	%ld",stateChangeMatrix[EXCLUSIVE][SHARED]);
	printf("\n10. number of shared to modified (SHD->MOD):	%ld",stateChangeMatrix[SHARED][MODIFIED]);
	printf("\n11. number of invalid to modified (INV->MOD):	%ld",stateChangeMatrix[INVALID][MODIFIED]);
	printf("\n12. number of exclusive to modified (EXC->MOD):	%ld",stateChangeMatrix[EXCLUSIVE][MODIFIED]);
	printf("\n13. number of owned to modified (OWN->MOD):	%ld",stateChangeMatrix[OWNER][MODIFIED]);
	printf("\n14. number of modified to owned (MOD->OWN):	%ld",stateChangeMatrix[MODIFIED][OWNER]);
	//printf("\n15. number of cache to cache transfers:		%ld",stateChangeMatrix[INVALID][SHARED]);
	printf("\n16. number of interventions:			%ld",interventions);
	printf("\n17. number of invalidations:			%ld",invalidations);
	printf("\n18. number of flushes:				%ld",flushes);
}

void Cache::setController(IMemoryController &memController)
{
	controller = &memController;
}

int Cache::setState(ulong addr,cacheState newState,bool isFlushNeeded)
{
	cacheState oldState = getState(addr);
	if (oldState == UNCACHED)
	{
		return -1;
	}
	else
	{
		setState(addr,newState);
		return 0;
	}
	recordStateChange(oldState,newState);
	if (newState == INVALID)
		invalidations++;
	else if (newState == SHARED && oldState != UNCACHED)
		interventions++;
	if (isFlushNeeded == true)
		flushes++;
}

bool Cache::hasLine(ulong addr)
{
	if (findLine(addr) == NULL) return false;
	else return true;
}

cacheState Cache::getState(ulong addr)
{
	cacheLine *temp;
	temp = findLine(addr);
	if (temp == NULL)
	{
		return UNCACHED;
	}
	else
	{
		return temp->getState();
	}
}

void Cache::recordStateChange(cacheState oldState,cacheState newState)
{
	stateChangeMatrix[oldState][newState]++;
}

void Cache::snoopBusTransaction(ulong addr,busTransaction transaction)
{
	//TODO: check flushOpts
	cacheState tempState = getState(addr);
	if (tempState != UNCACHED)
	{
		if (currentProtocol == MSI)
		{
			if (transaction == BUSRD)
			{
				if (tempState == MODIFIED)
				{
					setState(addr,SHARED,true);
				}
			}
			else if (transaction == BUSRDX)
			{
				if (tempState == SHARED)
				{
					setState(addr,INVALID,true);
				}
			}
		}
		else if (currentProtocol == MESI)
		{
			if (transaction == BUSRD)
			{
				if (tempState == MODIFIED || tempState == EXCLUSIVE)
				{
					setState(addr,SHARED,true);
				}
			}
			else if (transaction == BUSRDX)
			{
				if (tempState == SHARED || tempState == EXCLUSIVE || tempState == MODIFIED)
				{
					setState(addr,INVALID,true);
				}
			}
			else if (transaction == BUSUPGR)
			{
				if (tempState == SHARED)
				{
					setState(addr,INVALID);
				}
			}
		}
		else if (currentProtocol == MOESI)
		{
			//TODO
		}
	}
}
