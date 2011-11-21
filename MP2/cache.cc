/*******************************************************
cache.cc
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#include "Cache.h"

Cache::Cache(int s,int a,int b)
{
	reads = readMisses = writes = 0; 
	writeMisses = writeBacks = currentCycle = 0;


	size            = (ulong)(s);
	lineSize        = (ulong)(b);
	assoc           = (ulong)(a);
	sets            = (ulong)((s/b)/a);
	numLines        = (ulong)(s/b);
	log2Sets        = (ulong)(log2(int(sets)));
	log2Blk         = (ulong)(log2(int(b)));
	
	invalidations = interventions = flushes = transfers = 0;
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
	if(line == NULL)/*miss*/
	{
		if (op == 'w') 
		{
			writeMisses++;
			//printf("\n Write Miss: %x",addr);
			if (currentProtocol != MSI)
			{
				transfers += controller->requestBusTransaction(addr,BUSRDX,processorID);
			}
			else
			{
				controller->requestBusTransaction(addr,BUSRDX,processorID);
			}
			fillLine(addr,PRWR);
		}
		else if (op == 'r')
		{
			readMisses++;
			//printf("\n Read Miss: %ld",addr);
			if (currentProtocol != MSI)
			{
				transfers += controller->requestBusTransaction(addr,BUSRD,processorID);
			}
			else
			{
				controller->requestBusTransaction(addr,BUSRD,processorID);
			}
			fillLine(addr,PRRD);
		}
		
		//TODO:update cache to cache transfer counter.

	}
	else
	{
		updateLRU(line);
		cacheState currState = getState(addr);
		if(op == 'w')
		{
			if (currentProtocol == MSI)
			{
				if (currState == SHARED)
				{
					controller->requestBusTransaction(addr,BUSRDX,processorID);
					setState(addr,MODIFIED);
				}
				
			}
			else if (currentProtocol == MESI)
			{
				if (currState == SHARED)
				{
					controller->requestBusTransaction(addr,BUSUPGR,processorID);
				}
				setState(addr,MODIFIED);
			}
			else if (currentProtocol == MOESI)
			{
				if (currState == SHARED || currState == OWNER)
				{
					controller->requestBusTransaction(addr,BUSUPGR,processorID);
				}
				setState(addr,MODIFIED);
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
cacheLine *Cache::fillLine(ulong addr,processorAction action)
{ 
	ulong tag;
	cacheState newState;

	cacheLine *victim = findLineToReplace(addr);
	assert(victim != 0);

	//if(currentProtocol == MOESI)
	//{
	//	if (victim->getState() == OWNER)
	//	{
	//		//TODO:Check
	//		writeBack(addr);
	//		recordStateChange(OWNER,INVALID);
	//	}
	//}
	if (victim->getState() == MODIFIED)
	{
		//TODO:Check
		writeBack(addr);
	}

	tag = calcTag(addr);
	victim->setTag(tag);

	//TODO: write code for cache to cache transfers.
	if (action == PRRD)
	{
		if (currentProtocol == MSI)
		{
			newState = SHARED;
		}
		//For MESI and MOESI, EXCLUSIVE if copies do not exist;
		else if (controller->copiesExist(addr,processorID) == false)
		{
			newState = EXCLUSIVE;
		}
		else 
		{
			newState = SHARED;
		}
	}
	else if (action == PRWR)
	{
		newState = MODIFIED;
	}
	recordStateChange(INVALID,newState);
	victim->setState(newState);

	return victim;
}

void Cache::printStats()
{ 
	printf("\n01. number of reads:				%ld",reads);
	printf("\n02. number of read misses:			%ld",readMisses);
	printf("\n03. number of writes:				%ld",writes);
	printf("\n04. number of write misses:			%ld",writeMisses);
	printf("\n05. number of write backs:			%ld",writeBacks);
	printf("\n06. number of invalid to exclusive (INV->EXC):	%ld",stateChangeMatrix[INVALID][EXCLUSIVE]);
	printf("\n07. number of invalid to shared (INV->SHD):	%ld",stateChangeMatrix[INVALID][SHARED]);
	printf("\n08. number of modified to shared (MOD->SHD):	%ld",stateChangeMatrix[MODIFIED][SHARED]);
	printf("\n09. number of exclusive to shared (EXC->SHD):	%ld",stateChangeMatrix[EXCLUSIVE][SHARED]);
	printf("\n10. number of shared to modified (SHD->MOD):	%ld",stateChangeMatrix[SHARED][MODIFIED]);
	printf("\n11. number of invalid to modified (INV->MOD):	%ld",stateChangeMatrix[INVALID][MODIFIED]);
	printf("\n12. number of exclusive to modified (EXC->MOD):	%ld",stateChangeMatrix[EXCLUSIVE][MODIFIED]);
	printf("\n13. number of owned to modified (OWN->MOD):	%ld",stateChangeMatrix[OWNER][MODIFIED]);
	printf("\n14. number of modified to owned (MOD->OWN):	%ld",stateChangeMatrix[MODIFIED][OWNER]);
	printf("\n15. number of cache to cache transfers:		%d",transfers);
	printf("\n16. number of interventions:			%ld",interventions);
	printf("\n17. number of invalidations:			%ld",invalidations);
	printf("\n18. number of flushes:				%ld",flushes);
}

void Cache::setController(MemoryController &memController)
{
	controller = &memController;
}

int Cache::setState(ulong addr,cacheState newState,bool isFlushNeeded)
{
	cacheState oldState = getState(addr);
	cacheLine *currLine = findLine(addr);
	//if (oldState == INVALID)
	//{
		//return -1;
	//}
	recordStateChange(oldState,newState);
	if (newState == INVALID)
		invalidations++;
	else if (oldState == MODIFIED && (newState == OWNER || newState == SHARED))
		interventions++;
	if (isFlushNeeded == true)
		flushes++;
	currLine->setState(newState);
	return 0;
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
	cacheState state;
	if (temp == NULL)
	{
		state = INVALID;
	}
	else
	{
		state =  temp->getState();
	}
	return state;
}

void Cache::recordStateChange(cacheState oldState,cacheState newState)
{
	stateChangeMatrix[oldState][newState]++;
}

bool Cache::snoopBusTransaction(ulong addr,busTransaction transaction)
{
	//TODO: check flushOpts - not needed.
	cacheState tempState = getState(addr);
	bool flushNeeded = false;
	if (currentProtocol == MSI)
	{
		if (transaction == BUSRD)
		{
			if (tempState == MODIFIED)
			{
				flushNeeded = true;
				setState(addr,SHARED,flushNeeded);
			}
		}
		else if (transaction == BUSRDX)
		{
			if (tempState == SHARED)
			{
				setState(addr,INVALID);
			}
			else if (tempState == MODIFIED)
			{
				flushNeeded = true;
				setState(addr,INVALID,flushNeeded);
			}
		}
	}
	else if (currentProtocol == MESI)
	{
		if (transaction == BUSRD)
		{
			if (tempState == MODIFIED)
			{
				flushNeeded = true;
				setState(addr,SHARED,flushNeeded);
			}
			else if (tempState == EXCLUSIVE)
			{
				setState(addr,SHARED);
			}
		}
		else if (transaction == BUSRDX)
		{
			if (tempState == MODIFIED)
			{
				flushNeeded = true;
				setState(addr,INVALID,flushNeeded);
			}
			else if (tempState == SHARED || tempState == EXCLUSIVE)
			{
				setState(addr,INVALID);
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
		if (transaction == BUSRD)
		{
			if (tempState == MODIFIED)
			{
				flushNeeded = true;
				setState(addr,OWNER,flushNeeded);
			}
			else if (tempState == OWNER)
			{
				flushes++;
			}
			else if (tempState == EXCLUSIVE)
			{
				setState(addr,SHARED);
			}
		}
		else if (transaction == BUSRDX)
		{
			if (tempState != INVALID)
			{
				if (tempState != SHARED)
				{
					flushNeeded = true;
					setState(addr,INVALID,flushNeeded);
				}
				else
				{
					setState(addr,INVALID);
				}
			}
		}
		else if (transaction == BUSUPGR)
		{
			if (tempState == SHARED)
			{
				setState(addr,INVALID);
			}
			else if (tempState == OWNER)
			{
				setState(addr,INVALID);
			}
		}
	}
	return flushNeeded;
}

void Cache::init(coherenceProtocol protocol,int processorID)
{
	ulong i,j;
	for(i = 0;i <= INVALID; i++)
	{
		for (j = 0; j <= INVALID; j++)
		{
			stateChangeMatrix[i][j] = 0;
		}
	}


	tagMask =0;
	for(i=0;i<log2Sets;i++)
	{
		tagMask <<= 1;
		tagMask |= 1;
	}

	for(i=0; i<=sets; i++)
	{
		vector<cacheLine> tempCache;
		for(j=0; j<=assoc; j++) 
		{
			cacheLine tempBlock;
			tempBlock.invalidate();
			tempCache.push_back(tempBlock);
		}
		cache.push_back(tempCache);
	}
	//cache = new cacheLine*[sets];
	//for(i=0; i<sets; i++)
	//{
	//	cache[i] = new cacheLine[assoc];
	//	for(j=0; j<assoc; j++) 
	//	{
	//		cache[i][j].invalidate();
	//	}
	//}
	currentProtocol = protocol;
	this->processorID = processorID;
}
