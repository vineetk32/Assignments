/*******************************************************
cache.cc
Ahmad Samih & Yan Solihin
2009
{aasamih,solihin}@ece.ncsu.edu
********************************************************/

#include "Cache.h"


Cache::Cache(int s,int a,int b)
{
	ulong i, j;
	reads = readMisses = writes = 0; 
	writeMisses = writeBacks = currentCycle = 0;

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
			if (currentProtocol == MSI)
			{
				//newline->setState(MODIFIED);
				setState(addr,MODIFIED);
				controller->broadcastStateChange(addr,MODIFIED);
				//recordStateChange(INVALID,MODIFIED);
			}
			else 
			{
				if (controller->copiesExist(addr,processorID) == true)
				{
					//newline->setSeq(MODIFIED);
					setState(addr,MODIFIED);
					controller->broadcastStateChange(addr,MODIFIED);
					//recordStateChange(INVALID,MODIFIED);
				}
				else
				{
					//newline->setState(EXCLUSIVE);
					setState(addr,EXCLUSIVE);
					controller->broadcastStateChange(addr,EXCLUSIVE);
					//recordStateChange(INV
				}
			}
			
		}

	}
	else
	{
		/**since it's a hit, update LRU and update dirty flag**/
		updateLRU(line);
		if(op == 'w') line->setState(DIRTY);
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
	printf("\n06. number of invalid to exclusive (INV->EXC):	%d",stateChangeMatrix[INVALID][SHARED]);
	printf("\n07. number of invalid to shared (INV->SHD):	%d",stateChangeMatrix[INVALID][SHARED]);
	printf("\n08. number of modified to shared (MOD->SHD):	%d",stateChangeMatrix[MODIFIED][SHARED]);
	printf("\n09. number of exclusive to shared (EXC->SHD):	%d",stateChangeMatrix[EXCLUSIVE][SHARED]);
	printf("\n10. number of shared to modified (SHD->MOD):	%d",stateChangeMatrix[SHARED][MODIFIED]);
	printf("\n11. number of invalid to modified (INV->MOD):	%d",stateChangeMatrix[INVALID][MODIFIED]);
	printf("\n12. number of exclusive to modified (EXC->MOD):	%d",stateChangeMatrix[EXCLUSIVE][MODIFIED]);
	printf("\n13. number of owned to modified (OWN->MOD):	%d",stateChangeMatrix[OWNER][MODIFIED]);
	printf("\n14. number of modified to owned (MOD->OWN):	%d",stateChangeMatrix[MODIFIED][OWNER]);
	/*printf("\n15. number of cache to cache transfers:		%d",stateChangeMatrix[INVALID][SHARED]);
	printf("\n16. number of interventions:			0
	printf("\n17. number of invalidations:			21
	printf("\n18. number of flushes:				17*/
}

void Cache::setController(IMemoryController &memController)
{
	controller = &memController;
}

int Cache::setState(ulong addr,cacheState newState,bool isFlushNeeded)
{
	cacheLine *temp;
	temp = findLine(addr);
	if (temp == NULL)
	{
		return -1;
	}
	else
	{
		temp->setState(newState);
		return 0;
	}
	recordStateChange(temp->getState(),newState);
	//TODO: add interventions,invalidations,flushes
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
	//TODO: increment the flush counter everytime.
	cacheState tempState = getState(addr);
	if (tempState != UNCACHED)
	{
		if (currentProtocol == MSI)
		{
			if (transaction == BUSRD)
			{
				if (tempState == MODIFIED)
				{
					setState(addr,SHARED);
				}
			}
			else if (transaction == BUSRDX)
			{
				if (tempState == SHARED)
				{
					setState(addr,INVALID);
				}
			}
		}
		else if (currentProtocol == MESI)
		{
			if (transaction == BUSRD)
			{
				if (tempState == MODIFIED || tempState == EXCLUSIVE)
				{
					setState(addr,SHARED);
				}
			}
			else if (transaction == BUSRDX)
			{
				if (tempState == SHARED || tempState == EXCLUSIVE || tempState == MODIFIED)
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
		}
	}
}
