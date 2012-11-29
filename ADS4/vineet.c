#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define NUM_BITS 2097152
#define NUM_ARRAYS 8

#define hashsize(n) ((uint32_t)1<<(n))
#define hashmask(n) (hashsize(n)-1)

typedef struct bloom_filter
{
	unsigned char arrays[NUM_ARRAYS][NUM_BITS / 8];
} bf_t;

bf_t * create_bf()
{
	int i;
	bf_t *filter;
	filter = (bf_t *) malloc(sizeof(bf_t));
	for (i = 0; i < NUM_ARRAYS; i++)
	{
		memset(filter->arrays[i],0,NUM_BITS / 8);
	}
	return filter;
}


/*
-------------------------------------------------------------------------------
hashlittle() -- hash a variable-length key into a 32-bit value
  k       : the key (the unaligned variable-length array of bytes)
  length  : the length of the key, counting by bytes
  initval : can be any 4-byte value
Returns a 32-bit value.  Every bit of the key affects every bit of
the return value.  Two keys differing by one or two bits will have
totally different hash values.

The best hash table sizes are powers of 2.  There is no need to do
mod a prime (mod is sooo slow!).  If you need less than 32 bits,
use a bitmask.  For example, if you need only 10 bits, do
  h = (h & hashmask(10));
In which case, the hash table should have hashsize(10) elements.

If you are hashing n strings (uint8_t **)k, do it like this:
  for (i=0, h=0; i<n; ++i) h = hashlittle( k[i], len[i], h);

By Bob Jenkins, 2006.  bob_jenkins@burtleburtle.net.  You may use this
code any way you wish, private, educational, or commercial.  It's free.

Use for hash table lookup, or anything where one collision in 2^^32 is
acceptable.  Do NOT use for cryptographic purposes.
-------------------------------------------------------------------------------
*/

uint32_t hashlittle(const void *key, size_t length, uint32_t initval);

void insert_bf(bf_t *b, char *s)
{
	uint32_t hashval = 172341;
	int i,array_index,offset;
	for (i = 0; i < NUM_ARRAYS; i++)
	{
		hashval = hashlittle(s,strlen(s),hashval) & hashmask(21);
		offset = hashval % 8;
		array_index = (hashval - offset) / 8;
		//printf("\nHashval - %u",hashval);
		//printf("\nSetting %uth bit of %uth element of %uth array.",offset,array_index,i);
		//printf("\nBefore:%d\t",b->arrays[i][array_index]);
		b->arrays[i][array_index] = b->arrays[i][array_index] |  ( 1 << (int) offset );
		//printf("After:%d\n",b->arrays[i][array_index]);
	}
}

int is_element(bf_t *b, char *s)
{
	uint32_t hashval = 172341;
	int i,array_index,offset,count = 0;
	for (i = 0; i < NUM_ARRAYS; i++)
	{
		hashval = hashlittle(s,strlen(s),hashval) & hashmask(21);
		offset = hashval % 8;
		array_index = (hashval - offset) / 8;
		if (!(b->arrays[i][array_index] &  ( 1 << (int) offset )))
		{
			return 0;
		}
	}
	return 1;

}

//int main(int argc, char **argv)
//{
//	bf_t *filter;
//	filter = create_bf();
//	insert_bf(filter,argv[1]);
//	printf("\nIs element - %d\n",is_element(filter,argv[1]));
//	return 0;
//}

void sample_string_A(char *s, long i)
{  s[0] = (char)(1 + (i%254));
   s[1] = (char)(1 + ((i/254)%254));
   s[2] = (char)(1 + (((i/254)/254)%254));
   s[3] = 'a';
   s[4] = 'b';
   s[5] = (char)(1 + ((i*(i-3)) %217));
   s[6] = (char)(1 + ((17*i+129)%233 ));
   s[7] = '\0';
}
void sample_string_B(char *s, long i)
{  s[0] = (char)(1 + (i%254));
   s[1] = (char)(1 + ((i/254)%254));
   s[2] = (char)(1 + (((i/254)/254)%254));
   s[3] = 'a';
   s[4] = (char)(1 + ((i*(i-3)) %217));
   s[5] = (char)(1 + ((17*i+129)%233 ));
   s[6] = '\0';
}
void sample_string_C(char *s, long i)
{  s[0] = (char)(1 + (i%254));
   s[1] = (char)(1 + ((i/254)%254));
   s[2] = 'a';
   s[3] = (char)(1 + ((i*(i-3)) %217));
   s[4] = (char)(1 + ((17*i+129)%233 ));
   s[5] = '\0';
}
void sample_string_D(char *s, long i)
{  s[0] = (char)(1 + (i%254));
   s[1] = (char)(1 + ((i/254)%254));
   s[2] = (char)(1 + (((i/254)/254)%254));
   s[3] = 'b';
   s[4] = 'b';
   s[5] = (char)(1 + ((i*(i-3)) %217));
   s[6] = (char)(1 + ((17*i+129)%233 ));
   s[7] = '\0';
}
void sample_string_E(char *s, long i)
{  s[0] = (char)(1 + (i%254));
   s[1] = (char)(1 + ((i/254)%254));
   s[2] = (char)(1 + (((i/254)/254)%254));
   s[3] = 'a';
   s[4] = (char)(2 + ((i*(i-3)) %217));
   s[5] = (char)(1 + ((17*i+129)%233 ));
   s[6] = '\0';
}

int main()
{  long i,j; 
   bf_t * bloom;
   bloom = create_bf();
   printf("Created Filter\n");
   for( i= 0; i< 1450000; i++ )
   {  char s[8];
      sample_string_A(s,i);
      insert_bf( bloom, s );
   }
   for( i= 0; i< 500000; i++ )
   {  char s[7];
      sample_string_B(s,i);
      insert_bf( bloom, s );
   }
   for( i= 0; i< 50000; i++ )
   {  char s[6];
      sample_string_C(s,i);
      insert_bf( bloom, s );
   }
   printf("inserted 2,000,000 strings of length 8,7,6.\n");
   
   for( i= 0; i< 1450000; i++ )
   {  char s[8];
      sample_string_A(s,i);
      if( is_element( bloom, s ) != 1 )
	{  printf("found negative error (1)\n"); exit(0); }
   }
   for( i= 0; i< 500000; i++ )
   {  char s[7];
      sample_string_B(s,i);
      if( is_element( bloom, s ) != 1 )
	{  printf("found negative error (2)\n"); exit(0); }
   }
   for( i= 0; i< 50000; i++ )
   {  char s[6];
      sample_string_C(s,i);
      if( is_element( bloom, s ) != 1 )
	{  printf("found negative error (3)\n"); exit(0); }
   }
   j = 0;
   for( i= 0; i< 500000; i++ )
   {  char s[8];
      sample_string_D(s,i);
      if( is_element( bloom, s ) != 0 )
	j+=1;
   }
   for( i= 0; i< 500000; i++ )
   {  char s[7];
      sample_string_E(s,i);
      if( is_element( bloom, s ) != 0 )
	j+=1;
   }
   printf("Found %d positive errors out of 1,000,000 tests.\n",j);
   printf("Positive error rate %f\%.\n", (float)j/10000.0);

}
