#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 256

//#define STEP

typedef int key_t;
typedef int object_t;
typedef struct tr_n_t 
{
	key_t      key; 
	struct tr_n_t   *left;
	struct tr_n_t  *right;
	unsigned int height, measure, leftmin,rightmax;
	unsigned int leftInterval,rightInterval;
} m_tree_t;

typedef struct st_t
{
	m_tree_t *item;
	struct st_t *next;
} stack_t;

m_tree_t *currentblock = NULL;
int    size_left;
m_tree_t *free_list = NULL;
int nodes_taken = 0;
int nodes_returned = 0;


stack_t *get_node_st();
void return_node_st(stack_t *node);
stack_t *create_stack(void);
int stack_empty(stack_t *st);
void push( m_tree_t *x, stack_t *st);
m_tree_t *pop(stack_t *st);
m_tree_t *top_element(stack_t *st);
void remove_stack(stack_t *st);
void right_rotation(m_tree_t *n);
void left_rotation(m_tree_t *n);
void indent(char ch, int level);
void printTree (m_tree_t *root, int level);
m_tree_t *get_node();
void return_node(m_tree_t *node);
stack_t *get_node_st();
void return_node_st(stack_t *node);
m_tree_t *create_tree(void);
unsigned int setRightMax(m_tree_t *node);
unsigned int setLeftMin(m_tree_t *node);
unsigned int setMeasure(m_tree_t *node);
object_t *_delete_balanced(m_tree_t *tree, key_t delete_key);
void remove_tree(m_tree_t *tree);
m_tree_t *interval_find(m_tree_t *tree, key_t a, key_t b);
void check_tree( m_tree_t *tr, int depth, int lower, int upper );
int insert_balanced(m_tree_t *tree, key_t new_key,object_t *new_object,unsigned int leftMin, unsigned int rightMax);
unsigned int Max(unsigned int a, unsigned int b);
unsigned int Min(unsigned int a, unsigned int b);
void insert_interval(m_tree_t *tree, int a, int b);
void delete_interval(m_tree_t * tree, int a, int b);
int query_length(m_tree_t * tree);
void print_stack(stack_t *st);
void recursiveFixxer(m_tree_t *root);


stack_t *create_stack(void)
{
	stack_t *st;
	st = get_node_st();
	st->next = NULL;
	return( st );
}

int stack_empty(stack_t *st)
{
	return( st->next == NULL );
}

void push( m_tree_t *x, stack_t *st)
{
	stack_t *tmp;
	tmp = get_node_st();
	tmp->item = x;
	tmp->next = st->next;
	st->next = tmp;
}

m_tree_t *pop(stack_t *st)
{
	stack_t *tmp;
	m_tree_t *tmp_item;
	tmp = st->next;
	st->next = tmp->next;
	tmp_item = tmp->item;
	return_node_st( tmp );
	return( tmp_item );
}

m_tree_t *top_element(stack_t *st)
{
	return (st->next->item);
}

void remove_stack(stack_t *st)
{
	stack_t *tmp;
	do
	{
		tmp = st->next;
		return_node_st(st);
		st = tmp;
	}
	while ( tmp != NULL );
}

void right_rotation(m_tree_t *n)
{
	m_tree_t *tmp_node;
	key_t tmp_key;
	unsigned int leftKey, rightKey,leftInterval,rightInterval;
	tmp_node = n->right;

	tmp_key = n->key;
	leftKey = n->leftmin;
	rightKey = n->rightmax;
	leftInterval = n->leftInterval;
	rightInterval = n->rightInterval;

	n->right = n->left;
	n->key = n->left->key;
	n->leftmin = n->left->leftmin;
	n->rightmax = n->left->rightmax;
	n->leftInterval = n->left->leftInterval;
	n->rightInterval = n->left->rightInterval;

	n->left = n->right->left;
	n->right->left = n->right->right;
	n->right->right = tmp_node;

	n->right->key = tmp_key;
	n->right->leftmin = leftKey;
	n->right->rightmax = rightKey;
	n->right->leftInterval = leftInterval;
	n->right->rightInterval = rightInterval;
}

void left_rotation(m_tree_t *n)
{
	m_tree_t *tmp_node;
	key_t tmp_key;
	unsigned int leftKey, rightKey,leftInterval,rightInterval;

	tmp_node = n->left;
	tmp_key = n->key;
	leftKey = n->leftmin;
	rightKey = n->rightmax;
	leftInterval = n->leftInterval;
	rightInterval = n->rightInterval;


	n->left = n->right;
	n->key = n->right->key;
	n->leftmin =  n->right->leftmin;
	n->rightmax = n->right->rightmax;
	n->leftInterval = n->right->leftInterval;
	n->rightInterval = n->right->rightInterval;

	n->right = n->left->right;
	n->left->right = n->left->left;
	n->left->left = tmp_node;

	n->left->key = tmp_key;
	n->left->leftmin = leftKey;
	n->left->rightmax = rightKey;
	n->left->leftInterval = leftInterval;
	n->left->rightInterval = rightInterval;

}

void indent(char ch, int level)
{
	int i;
	for (i = 0; i < level; i++ )
		putchar(ch);
}

void printTree (m_tree_t *root, int level)
{
	if (root->right == NULL)
	{
		indent('\t', level);
		printf("%d ]%d,%d[ (%d,%d) (%d)\n", root->key,root->leftInterval,root->rightInterval,root->leftmin,root->rightmax,root->measure);
	}
	else
	{
		printTree(root->right, level + 1);
		indent('\t', level);
		printf("%d ]%d,%d[ (%d,%d) (%d)\n", root->key,root->leftInterval,root->rightInterval,root->leftmin,root->rightmax,root->measure);
		printTree(root->left, level + 1);
	}
}

m_tree_t *get_node()
{ 
	m_tree_t *tmp;
	nodes_taken += 1;
	if( free_list != NULL )
	{  
		tmp = free_list;
		free_list = free_list -> right;
	}
	else
	{  
		if( currentblock == NULL || size_left == 0)
		{  
			currentblock = (m_tree_t *) malloc( BLOCKSIZE * sizeof(m_tree_t) );
			size_left = BLOCKSIZE;
		}
		tmp = currentblock++;
		size_left -= 1;
	}
	return( tmp );
}

void return_node(m_tree_t *node)
{  
	node->right = free_list;
	free_list = node;
	nodes_returned +=1;
}

stack_t *get_node_st()
{ 
	stack_t *tmp;
	tmp = (stack_t *) malloc(sizeof(stack_t));
	return( tmp );
}
void return_node_st(stack_t *node)
{
	free(node);
}

m_tree_t *create_tree(void)
{
	m_tree_t *tmp_node;
	tmp_node = get_node();
	tmp_node->left = NULL;
	return( tmp_node );
}

object_t *_delete_balanced(m_tree_t *tree, key_t delete_key)
{
	m_tree_t *tmp_node, *upper_node, *other_node;
	int finished;
	stack_t *stack;
	object_t *deleted_object;
	if( tree->left == NULL )
		return( NULL );
	else if( tree->right == NULL )
	{  
		if(  tree->key == delete_key )
		{  
			deleted_object = (object_t *) tree->left;
			tree->left = NULL;
			return( deleted_object );
		}
		else
			return( NULL );
	}
	else
	{
		stack = create_stack();
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			push(tmp_node,stack);
			upper_node = tmp_node;
			if( delete_key < tmp_node->key )
			{  
				tmp_node   = upper_node->left; 
				other_node = upper_node->right;
			} 
			else
			{  
				tmp_node   = upper_node->right; 
				other_node = upper_node->left;
			} 
		}
		if( tmp_node->key != delete_key )
			return( NULL );
		else
		{  
			upper_node->key   = other_node->key;
			upper_node->left  = other_node->left;
			upper_node->right = other_node->right;
			deleted_object = (object_t *) tmp_node->left;
			return_node( tmp_node );
			return_node( other_node );
			return( deleted_object );
		}
		/* rebalance */
		//finished = 0;
		while( !stack_empty(stack))// && !finished )
		{
			int tmp_height, old_height;
			tmp_node = pop(stack);
			old_height= tmp_node->height;
			if( tmp_node->left->height - tmp_node->right->height == 2 )
			{
				if( tmp_node->left->left->height - tmp_node->right->height == 1 )
				{
					right_rotation(tmp_node);
					tmp_node->right->height = tmp_node->right->left->height + 1;
					tmp_node->height = tmp_node->right->height + 1;
				}
				else
				{
					left_rotation(tmp_node->left);
					right_rotation(tmp_node);
					tmp_height = tmp_node->left->left->height;
					tmp_node->left->height = tmp_height + 1;
					tmp_node->right->height = tmp_height + 1;
					tmp_node->height = tmp_height + 2;
				}
			}
			else if( tmp_node->left->height - tmp_node->right->height == -2 )
			{
				if( tmp_node->right->right->height - tmp_node->left->height == 1 )
				{
					left_rotation( tmp_node );
					tmp_node->left->height = tmp_node->left->right->height + 1;
					tmp_node->height = tmp_node->left->height + 1;
				}
				else
				{
					right_rotation( tmp_node->right );
					left_rotation( tmp_node );
					tmp_height = tmp_node->right->right->height;
					tmp_node->left->height = tmp_height + 1;
					tmp_node->right->height = tmp_height + 1;
					tmp_node->height = tmp_height + 2;
				}
			}
			else /* update height even if there
				 was no rotation */
			{
				if( tmp_node->left->height > tmp_node->right->height )
					tmp_node->height = tmp_node->left->height + 1;
				else
					tmp_node->height = tmp_node->right->height + 1;
			}
			/*if( tmp_node->height == old_height )
			finished = 1;*/
		}
		remove_stack(stack);
	}
}

void remove_tree(m_tree_t *tree)
{
	m_tree_t *current_node, *tmp;
	if( tree->left == NULL )
		return_node( tree );
	else
	{  
		current_node = tree;
		while(current_node->right != NULL )
		{  
			if( current_node->left->right == NULL )
			{  
				return_node( current_node->left );
				tmp = current_node->right;
				return_node( current_node );
				current_node = tmp;
			}
			else
			{  
				tmp = current_node->left;
				current_node->left = tmp->right;
				tmp->right = current_node; 
				current_node = tmp;
			}
		}
		return_node( current_node );
	}
}

m_tree_t *interval_find(m_tree_t *tree, key_t a, key_t b)
{ 
	m_tree_t *tr_node;
	m_tree_t *node_stack[200]; int stack_p = 0;
	m_tree_t *result_list, *tmp;
	result_list = NULL;
	node_stack[stack_p++] = tree;
	while( stack_p > 0 )
	{  
		tr_node = node_stack[--stack_p];
		if( tr_node->right == NULL )
		{ 
			/* reached leaf, now test */
			if( a <= tr_node->key && tr_node->key < b )
			{  
				tmp = get_node();        /* leaf key in interval */
				tmp->key  = tr_node->key; /* copy to output list */  
				tmp->left = tr_node->left;   
				tmp->right = result_list;
				result_list = tmp;
			}
		} /* not leaf, might have to follow down */
		else if ( b <= tr_node->key ) /* entire interval left */
			node_stack[stack_p++] = tr_node->left;
		else if ( tr_node->key <= a ) /* entire interval right*/
			node_stack[stack_p++] = tr_node->right;
		else   /* node key in interval, follow left and right */
		{  
			node_stack[stack_p++] = tr_node->left;
			node_stack[stack_p++] = tr_node->right;
		}
	}
	return( result_list );
}

void check_tree( m_tree_t *tr, int depth, int lower, int upper )
{
	if ( tr->left == NULL )
	{
		printf("Tree Empty\n"); return;
	}
	if ( tr->key < lower || tr->key >= upper )
		printf("Wrong Key Order \n");
	if ( tr->right == NULL )
	{
		if( *((int *) tr->left) == 10*tr->key + 2 )
			printf("%d (%d)  ", tr->key, depth );
		else
			printf("Wrong Object \n");
	}
	else
	{
		check_tree(tr->left, depth+1, lower, tr->key ); 
		check_tree(tr->right, depth+1, tr->key, upper ); 
	}
}

int insert_balanced(m_tree_t *tree, key_t new_key,object_t *new_object,unsigned int leftInterval, unsigned int rightInterval)
{
	m_tree_t *tmp_node;
	stack_t *stack;
	unsigned int currLeftInterval, currRightInterval;
	currLeftInterval = leftInterval;
	currRightInterval = rightInterval;
	if( tree->left == NULL )
	{
		tree->left = (m_tree_t *) new_object;
		tree->key = new_key;
		tree->height = 0;
		tree->right = NULL;
		tree->measure = 1;
		tree->leftmin = currLeftInterval;
		tree->rightmax = currRightInterval;
		tree->leftInterval = currLeftInterval;
		tree->rightInterval = currRightInterval;
	}
	else
	{
		stack = create_stack();
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			tmp_node->leftInterval = Min(tmp_node->leftInterval, currLeftInterval);
			tmp_node->rightInterval = Max(tmp_node->rightInterval,currRightInterval);

			push(tmp_node,stack);
			if( new_key < tmp_node->key )
			{
				currRightInterval = tmp_node->key;
				tmp_node = tmp_node->left;
			}
			else
			{
				currLeftInterval = tmp_node->key;
				tmp_node = tmp_node->right;
			}
		}
		/* found the candidate leaf. Test whether key distinct */
		if( tmp_node->key == new_key )
		{
			//printf("\nOverlapping interval - %d,%d",leftMin,rightMax);
			tmp_node->leftmin = Min(leftInterval,tmp_node->leftmin);
			tmp_node->rightmax = Max(rightInterval,tmp_node->rightmax);

			tmp_node->rightInterval = currRightInterval;
			tmp_node->leftInterval = currLeftInterval;
		}
		else 
		{
			m_tree_t *old_leaf, *new_leaf;

			old_leaf = get_node();
			old_leaf->left = tmp_node->left;
			old_leaf->key = tmp_node->key;
			old_leaf->leftmin = tmp_node->leftmin;
			old_leaf->rightmax = tmp_node->rightmax;
			old_leaf->measure = 1;
			old_leaf->height = 0;
			old_leaf->right= NULL;

			new_leaf = get_node();
			new_leaf->left = (m_tree_t *) new_object;
			new_leaf->key = new_key;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			new_leaf->measure = 1;
			new_leaf->leftmin = leftInterval;
			new_leaf->rightmax = rightInterval;
			if( tmp_node->key < new_key )
			{
				tmp_node->left = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = new_key;

				old_leaf->rightInterval = new_key;
				old_leaf->leftInterval = currLeftInterval;

				new_leaf->rightInterval = currRightInterval;
				new_leaf->leftInterval = new_key;

			}
			else
			{
				tmp_node->left = new_leaf;
				tmp_node->right = old_leaf;

				new_leaf->rightInterval = new_key;
				new_leaf->leftInterval = currLeftInterval;

				old_leaf->rightInterval = currRightInterval;
				old_leaf->leftInterval = new_key;

			}
			tmp_node->height = 1;
			tmp_node->rightmax = setRightMax(tmp_node);
			tmp_node->leftmin = setLeftMin(tmp_node);
			tmp_node->measure = tmp_node->right->key - tmp_node->left->key;
			tmp_node->leftInterval = currLeftInterval;
			tmp_node->rightInterval = currRightInterval;
			//tmp_node->measure = setMeasure(tmp_node);
			//push(tmp_node,stack);
		}
		/* rebalance */
		while( !stack_empty(stack))
		{
			int tmp_height, old_height;
			//print_stack(stack);
			tmp_node = pop(stack);
			old_height= tmp_node->height;
			if( tmp_node->left->height - tmp_node->right->height == 2 )
			{
				if( tmp_node->left->left->height -
					tmp_node->right->height == 1 )
				{
					right_rotation( tmp_node );
					tmp_node->right->height = tmp_node->right->left->height + 1;
					tmp_node->height = tmp_node->right->height + 1;
				}
				else
				{
					left_rotation( tmp_node->left );
					right_rotation( tmp_node );
					tmp_height = tmp_node->left->left->height;
					tmp_node->left->height = tmp_height + 1;
					tmp_node->right->height = tmp_height + 1;
					tmp_node->height = tmp_height + 2;
				}
			}
			else if( tmp_node->left->height - tmp_node->right->height == -2 )
			{
				if( tmp_node->right->right->height - tmp_node->left->height == 1 )
				{
					left_rotation( tmp_node );
					tmp_node->left->height = tmp_node->left->right->height + 1;
					tmp_node->height = tmp_node->left->height + 1;
				}
				else
				{
					right_rotation( tmp_node->right );
					left_rotation( tmp_node );
					tmp_height = tmp_node->right->right->height;
					tmp_node->left->height = tmp_height + 1;
					tmp_node->right->height = tmp_height + 1;
					tmp_node->height = tmp_height + 2;
				}
			}
			else
			{
				if( tmp_node->left->height > tmp_node->right->height )
					tmp_node->height = tmp_node->left->height + 1;
				else
					tmp_node->height = tmp_node->right->height + 1;
			}
			tmp_node->rightmax = setRightMax(tmp_node);
			tmp_node->leftmin = setLeftMin(tmp_node);
			tmp_node->measure = setMeasure(tmp_node);
		}
		/*while( !stack_empty(stack))
		{
			tmp_node = pop(stack);
			tmp_node->rightmax = setRightMax(tmp_node);
			tmp_node->leftmin = setLeftMin(tmp_node);
			tmp_node->measure = setMeasure(tmp_node);
		}*/
		remove_stack(stack);
	}
	tree->leftInterval = tree->leftmin;
	tree->rightInterval = tree->rightmax;
	recursiveFixxer(tree);
	return( 0 );
}
#ifdef STEP
int main()
{
	m_tree_t *searchtree;
	char nextop;
	searchtree = create_tree();
	printf("Made Tree\n");
	printf("In the following, the key n is associated wth the object 10n+2\n");
	while ((nextop = getchar()) != 'q')
	{ 
		if ( nextop == 'i' )
		{ 
			int leftKey,rightKey;
			scanf(" %d,%d", &leftKey,&rightKey);
			insert_interval(searchtree,leftKey,rightKey);
		}
		else if ( nextop == 'v' )
		{ 
			int a, b;  m_tree_t *results, *tmp;
			scanf(" %d %d", &a, &b);
			results = interval_find( searchtree, a, b );
			if( results == NULL )
				printf("  no keys found in the interval [%d,%d[\n", a, b);
			else
			{
				printf("  the following keys found in the interval [%d,%d[\n", a, b);
				while( results != NULL )
				{  
					printf("(%d,%d) ", results->key, *((int *) results->left) );
					tmp = results;
					results = results->right;
					return_node( tmp );
				}
				printf("\n");
			}
		}
		else if ( nextop == 'd' )
		{ 
			int delkey, *delobj;
			scanf(" %d", &delkey);
			delobj = _delete_balanced( searchtree, delkey);
			if( delobj == NULL )
				printf("  delete failed for key %d\n", delkey);
			else
				printf("  delete successful, deleted object %d for key %d\n", *delobj, delkey);
		}
		else if ( nextop == '?' )
		{  
			printf("  Checking tree\n"); 
			check_tree(searchtree,0,-1000,1000);
			printf("\n");
			if( searchtree->left != NULL )
				printf("key in root is %d\n",	 searchtree->key );
			printf("  Finished Checking tree\n"); 
		}
		else if (nextop == 'p')
		{
			printf("\nPrinting Tree - \n");
			printTree(searchtree,0);
		}
	}

	remove_tree( searchtree );
	printf("Removed tree.\n");
	return(0);
}
#endif

unsigned int Max(unsigned int a, unsigned int b)
{
	if (a > b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

unsigned int Min(unsigned int a, unsigned int b)
{
	if (a < b)
	{
		return a;
	}
	else
	{
		return b;
	}
}

unsigned int setRightMax(m_tree_t *node)
{
	return Max(node->left->rightmax, node->right->rightmax);
}

unsigned int setLeftMin(m_tree_t *node)
{
	return Min(node->left->leftmin, node->right->leftmin);
}

unsigned int setMeasure(m_tree_t *node)
{
	unsigned int l = node->leftInterval;
	unsigned int r = node->rightInterval;
	unsigned int measure = 1;
	//if (node->height > 1 )
	//{
		if (node->right->leftmin < l && node->left->rightmax >= r)
		{
			measure = r - l;
		}
		else if (node->right->leftmin >= l && node->left->rightmax >= r)
		{
			measure = (r - node->key) + node->left->measure;
		}
		else if (node->right->leftmin < l && node->left->rightmax < r)
		{
			measure = node->right->measure + (node->key - l);
		}
		else if (node->right->leftmin >= l && node->left->rightmax < r)
		{
			measure = node->right->measure + node->left->measure;
		}
		else
		{
			printf("\nNone of the 4 cases!");
		}
	//}
	//else
	//{
	//	measure = node->right->rightInterval - node->left->leftInterval;
	//}
	return measure;
}

#ifndef STEP
int main()
{
	int i; 
	m_tree_t *t;
	printf("starting \n");
	t = create_tree();
	for(i=0; i< 50; i++ )
	{
		insert_interval( t, 2*i, 2*i +1 );
	}
	printf("inserted first 50 intervals, total length is %d, should be 50.\n", query_length(t));
	insert_interval( t, 0, 100 );
	printf("inserted another interval, total length is %d, should be 100.\n", query_length(t));
	for(i=1; i< 50; i++ )
	{
		insert_interval( t, 199 - (3*i), 200 ); // [52,200] is longest
	}
	printf("inserted further 49 intervals, total length is %d, should be 200.\n", query_length(t));
	for(i=2; i< 50; i++ )
	{
		delete_interval(t, 2*i, 2*i +1 );
	}
	delete_interval(t,0,100);
	printf("deleted some intervals, total length is %d, should be 150.\n", query_length(t));
	insert_interval( t, 1,2 ); 
	for(i=49; i>0; i-- )
	{
		delete_interval( t, 199 - (3*i), 200 ); 
	}
	insert_interval( t, 0,2 );
	insert_interval( t, 1,5 );  
	printf("deleted some intervals, total length is %d, should be 5.\n", query_length(t));
	insert_interval( t, 0, 100 );
	printf("inserted another interval, total length is %d, should be 100.\n", query_length(t));
	for(i=0; i<=3000; i++ )
	{
		insert_interval( t, 2000+i, 3000+i ); 
	}
	printf("inserted 3000 intervals, total length is %d, should be 4100.\n", query_length(t));
	for(i=0; i<=3000; i++ )
	{
		delete_interval( t, 2000+i, 3000+i ); 
	}
	printf("deleted 3000 intervals, total length is %d, should be 100.\n", query_length(t));
	for(i=0; i<=100; i++ )
	{
		insert_interval( t, 10*i, 10*i+100 ); 
	}
	printf("inserted another 100 intervals, total length is %d, should be 1100.\n", query_length(t));
	delete_interval( t, 1,2 ); 
	delete_interval( t, 0,2 ); 
	delete_interval( t, 2,3 ); 
	delete_interval( t, 0,1 ); 
	delete_interval( t, 1,5 );
	printf("deleted some intervals, total length is %d, should be still 1100.\n", query_length(t)); 
	for(i=0; i<= 100; i++ )
	{
		delete_interval(t, 10*i, 10*i+100);
	}
	delete_interval(t,0,100);
	printf("deleted last interval, total length is %d, should be 0.\n", query_length(t));
	for( i=0; i<100000; i++)
	{
		insert_interval(t, i, 1000000);
	}
	printf("inserted again 100000 intervals, total length is %d, should be 1000000.\n", query_length(t));
	printf("End Test\n");
}
#endif

int query_length(m_tree_t * tree)
{
	return tree->measure;
}

void insert_interval(m_tree_t *tree, int leftKey, int rightKey)
{
	int insobj;
	insobj = 10*leftKey+2;
	insert_balanced(tree, leftKey, &insobj,leftKey,rightKey );
	insobj = 10*rightKey+2;
	insert_balanced(tree, rightKey, &insobj,leftKey,rightKey );
	//recursiveFixxer(tree);
}
void delete_interval(m_tree_t * tree, int a, int b)
{
}

void print_stack(stack_t *st)
{
	stack_t *top;
	top = st;
	printf("\nStack contents - ");
	do 
	{
		top = top->next;
		printf(" %d",top->item->key);
	} while (top->next != NULL);
	printf(" EOS.\n");
}

void recursiveFixxer(m_tree_t *root)
{
	if (root->height > 1)
	{
		root->left->leftInterval = root->leftInterval;
		root->left->rightInterval = root->key;

		root->right->rightInterval = root->rightInterval;
		root->right->leftInterval = root->key;

		recursiveFixxer(root->left);
		recursiveFixxer(root->right);
		root->rightmax = setRightMax(root);
		root->leftmin =  setLeftMin(root);
		root->measure =  setMeasure(root);
	}
	else if (root->height == 1)
	{
		root->rightmax = setRightMax(root);
		root->leftmin =  setLeftMin(root);

		//Who will fix the fixxer?
		if (root->leftInterval < root->leftmin)
			root->leftInterval = root->leftmin;

		if (root->rightInterval > root->rightmax)
			root->leftInterval = root->rightmax;

		root->measure =  setMeasure(root);

		root->left->leftInterval = root->leftInterval;
		root->left->rightInterval = root->key;

		root->right->rightInterval = root->rightInterval;
		root->right->leftInterval = root->key;

	}
}

