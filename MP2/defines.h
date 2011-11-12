#ifndef DEFINES_H
#define DEFINES_H

inline long log2(int x) { return long(  log(double(x)) / log (double(2))); }

typedef unsigned long ulong;
typedef unsigned char uchar;
typedef unsigned int uint;

enum cacheState {
	//Should invalid and uncached be treated the same?
	UNCACHED = -1,
	MODIFIED = 0,
	OWNER,
	EXCLUSIVE,
	SHARED,
	INVALID
};

enum coherenceProtocol {
	MSI = 0,
	MESI,
	MOESI
};

enum busTransaction {
	BUSRD = 0,
	BUSRDX,
	BUSUPGR,
	FLUSH,
	FLUSHOPT,
	FLUSHWB
};

class cacheLine 
{
protected:
	ulong tag;
	cacheState state;
	ulong seq;


public:
	cacheLine()                          { tag = 0; state = UNCACHED; }
	ulong getTag()                       { return tag; }
	cacheState getState()                { return state;}
	ulong getSeq()                       { return seq; }
	void setSeq(ulong Seq)               { seq = Seq;}
	void setState(cacheState newState)   { state = newState;}
	void setTag(ulong a)                 { tag = a; }
	void invalidate()                    { tag = 0; state = INVALID; }//useful function
	bool isValid()                       { return ((state) != INVALID); }
};


class MemoryBlock {
private:
	//ulong addr;
	int ownerProcessor;
	cacheState state;
	
public:
	MemoryBlock()                         { ownerProcessor = -1; state = UNCACHED; }
	void setState(cacheState newState)    { state = newState; }
	cacheState getState()                 { return state; }
	
};

#endif
