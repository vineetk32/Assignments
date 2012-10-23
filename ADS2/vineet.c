#include <stdio.h>
#include <stdlib.h>

#define BLOCKSIZE 256

#define STEP

typedef int key_t;
typedef int object_t;

typedef struct interval_node_t
{
	unsigned int leftInterval, rightInterval;
	struct interval_node_t *next;
} interval_node;

typedef struct tr_n_t 
{
	key_t      key; 
	struct tr_n_t   *left;
	struct tr_n_t  *right;
	unsigned int height, measure, leftmin,rightmax;
	unsigned int leftInterval,rightInterval;
	interval_node *list;
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
object_t *_delete_balanced(m_tree_t *tree, key_t delete_key,unsigned int leftMin, unsigned int rightMax);
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
interval_node *addToList(interval_node *list, unsigned int leftInterval,unsigned int rightInterval);
interval_node  *deleteFromList(interval_node *list, unsigned int leftInterval,unsigned int rightInterval);
int intervalInList(interval_node *list,unsigned int leftInterval,unsigned int rightInterval);
void freeList(interval_node *list);
unsigned int getSmallest(interval_node *list);
unsigned int getLargest(interval_node *list);
void changeIntervals(m_tree_t *node,unsigned int leftInterval,unsigned int rightInterval);
int isInList(interval_node *list,unsigned int key);
void printTreeList(m_tree_t *node);
void printList(interval_node *node);
void fixList(unsigned int key,interval_node *this_list,interval_node *other_list);
interval_node *cloneList(interval_node *list);
interval_node *appendList(interval_node *list,interval_node *toAppend);

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
	//interval_node *list;

	tmp_node = n->right;
	tmp_key = n->key;
	leftKey = n->leftmin;
	rightKey = n->rightmax;
	leftInterval = n->leftInterval;
	rightInterval = n->rightInterval;
	//list = cloneList(n->list);

	n->right = n->left;
	n->key = n->left->key;
	n->leftmin = n->left->leftmin;
	n->rightmax = n->left->rightmax;
	n->leftInterval = n->left->leftInterval;
	n->rightInterval = n->left->rightInterval;
	//n->list = appendList(n->list,n->left->list);

	n->left = n->right->left;
	n->right->left = n->right->right;
	n->right->right = tmp_node;

	n->right->key = tmp_key;
	n->right->leftmin = leftKey;
	n->right->rightmax = rightKey;
	n->right->leftInterval = leftInterval;
	n->right->rightInterval = rightInterval;
	//n->right->list = appendList(n->right->list,list);
	//printf("\nnl - %p, nll - %p, nrl - %p",(void *) &(n->list), (void *) &(n->left->list), (void *) &(n->right->list));
	//freeList(list);
}

void left_rotation(m_tree_t *n)
{
	m_tree_t *tmp_node;
	key_t tmp_key;
	unsigned int leftKey, rightKey,leftInterval,rightInterval;
	//interval_node *list;

	tmp_node = n->left;
	tmp_key = n->key;
	leftKey = n->leftmin;
	rightKey = n->rightmax;
	leftInterval = n->leftInterval;
	rightInterval = n->rightInterval;
	//list = cloneList(n->list);


	n->left = n->right;
	n->key = n->right->key;
	n->leftmin =  n->right->leftmin;
	n->rightmax = n->right->rightmax;
	n->leftInterval = n->right->leftInterval;
	n->rightInterval = n->right->rightInterval;
	//n->list =appendList(n->list, n->right->list);

	n->right = n->left->right;
	n->left->right = n->left->left;
	n->left->left = tmp_node;

	n->left->key = tmp_key;
	n->left->leftmin = leftKey;
	n->left->rightmax = rightKey;
	n->left->leftInterval = leftInterval;
	n->left->rightInterval = rightInterval;
	//n->left->list = appendList(n->left->list,list);
	//printf("\nnl - %lx, nll - %lx, nrl - %lx",(unsigned long int) &(n->list), (unsigned long int) &(n->left->list), (unsigned long int) &(n->right->list));
	//freeList(list);
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
	tmp->list = NULL;
	tmp->left = NULL;
	tmp->right = NULL;
	tmp->height = 0;
	tmp->leftInterval = 0;
	tmp->rightInterval = 0;
	tmp->measure = 0;
	tmp->leftmin = 0;
	tmp->rightmax = 0;
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
	tmp_node->left = NULL;
	return( tmp_node );
}

object_t *_delete_balanced(m_tree_t *tree, key_t delete_key,unsigned int leftInterval, unsigned int rightInterval)
{
	m_tree_t *tmp_node, *upper_node, *other_node;
	unsigned int currLeftInterval,currRightInterval;
	stack_t *stack;
	object_t *deleted_object;
	currLeftInterval = leftInterval;
	currRightInterval = rightInterval;
	if( tree->left == NULL )
		return( NULL );
	else if( tree->right == NULL )
	{
		if(  tree->key == delete_key )
		{
			deleted_object = (object_t *) tree->left;
			tree->left = NULL;
			freeList(tree->list);
			tree->list = NULL;
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
			//printf("\nState of lists at this stage - ");
			//printTreeList(tmp_node);
			push(tmp_node,stack);
			upper_node = tmp_node;
			if (delete_key >= currLeftInterval && delete_key <= currRightInterval)
			{
				if (intervalInList(tmp_node->list,leftInterval,rightInterval))
				{
					tmp_node->list = deleteFromList(tmp_node->list,leftInterval,rightInterval);
					changeIntervals(tmp_node,leftInterval,rightInterval);
				}
			}
			if( delete_key < tmp_node->key )
			{
				tmp_node = upper_node->left; 
				currRightInterval = tmp_node->key;
				other_node = upper_node->right;
			}
			else
			{
				currLeftInterval = tmp_node->key;
				tmp_node   = upper_node->right; 
				other_node = upper_node->left;
			}
		}
		if (intervalInList(tmp_node->list,leftInterval,rightInterval))
		{
			tmp_node->list = deleteFromList(tmp_node->list,leftInterval,rightInterval);
			changeIntervals(tmp_node,leftInterval,rightInterval);
		}

		if( tmp_node->key != delete_key )
			return( NULL );
		else if (!isInList(tmp_node->list,delete_key))
		{
			upper_node->key   = other_node->key;
			upper_node->left  = other_node->left;
			upper_node->right = other_node->right;
			upper_node->height = 0;
			freeList(tmp_node->list);
			tmp_node->list = NULL;
			return_node( tmp_node );
			freeList(other_node->list);
			other_node->list = NULL;
			return_node( other_node );
		}
		/* rebalance */
		pop(stack);
		while( !stack_empty(stack))
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
			tmp_node->rightmax = setRightMax(tmp_node);
			tmp_node->leftmin = setLeftMin(tmp_node);
			tmp_node->measure = setMeasure(tmp_node);
		}
		remove_stack(stack);
	}
	tree->leftInterval = tree->leftmin;
	tree->rightInterval = tree->rightmax;
	recursiveFixxer(tree);
}

void remove_tree(m_tree_t *tree)
{
	m_tree_t *current_node, *tmp;
	if( tree->left == NULL )
	{
		freeList(tree->list);
		tree->list = NULL;
		return_node( tree );
	}
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
		freeList(current_node->list);
		current_node->list = NULL;
		return_node( current_node );
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
		tree->list = NULL;
		tree->list = addToList(tree->list,leftInterval,rightInterval);
	}
	else
	{
		stack = create_stack();
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			tmp_node->leftInterval = Min(tmp_node->leftInterval, currLeftInterval);
			tmp_node->rightInterval = Max(tmp_node->rightInterval,currRightInterval);

			if (tmp_node->key >= currLeftInterval && tmp_node->key <= currRightInterval)
				tmp_node->list = addToList(tmp_node->list,currLeftInterval,currRightInterval);
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
		if( tmp_node->key == new_key )
		{
			tmp_node->leftmin = Min(leftInterval,tmp_node->leftmin);
			tmp_node->rightmax = Max(rightInterval,tmp_node->rightmax);

			tmp_node->rightInterval = currRightInterval;
			tmp_node->leftInterval = currLeftInterval;
			if (intervalInList(tmp_node->list,leftInterval,rightInterval) != 0)
			{
				tmp_node->list = addToList(tmp_node->list,leftInterval,rightInterval);
			}
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
			old_leaf->list = cloneList(tmp_node->list);
			old_leaf->right= NULL;

			new_leaf = get_node();
			new_leaf->left = (m_tree_t *) new_object;
			new_leaf->key = new_key;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			new_leaf->measure = 1;
			new_leaf->list = NULL;
			new_leaf->leftmin = leftInterval;
			new_leaf->rightmax = rightInterval;
			if( tmp_node->key < new_key )
			{
				tmp_node->left = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = new_key;

				old_leaf->rightInterval = new_key;
				old_leaf->leftInterval = currLeftInterval;

				if (!intervalInList(old_leaf->list,currLeftInterval,new_key))
				{
					old_leaf->list = addToList(old_leaf->list,currLeftInterval,new_key);
				}

				new_leaf->rightInterval = currRightInterval;
				new_leaf->leftInterval = new_key;

				if (!intervalInList(new_leaf->list,new_key,currRightInterval))
				{
					new_leaf->list = addToList(new_leaf->list,new_key,currRightInterval);
				}


			}
			else
			{
				tmp_node->left = new_leaf;
				tmp_node->right = old_leaf;

				new_leaf->rightInterval = new_key;
				new_leaf->leftInterval = currLeftInterval;

				old_leaf->rightInterval = currRightInterval;
				old_leaf->leftInterval = new_key;

				if (!intervalInList(old_leaf->list,new_key,currRightInterval))
				{
					old_leaf->list = addToList(old_leaf->list,new_key,currRightInterval);
				}


				if (!intervalInList(new_leaf->list,currLeftInterval,new_key))
				{
					new_leaf->list = addToList(new_leaf->list,currLeftInterval,new_key);
				}

			}
			tmp_node->height = 1;
			tmp_node->rightmax = setRightMax(tmp_node);
			tmp_node->leftmin = setLeftMin(tmp_node);
			tmp_node->measure = tmp_node->right->key - tmp_node->left->key;
			tmp_node->leftInterval = currLeftInterval;
			tmp_node->rightInterval = currRightInterval;
			if (!intervalInList(tmp_node->list,currLeftInterval,currRightInterval))
			{
				tmp_node->list = addToList(tmp_node->list,currLeftInterval,currRightInterval);
			}

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
		else if ( nextop == 'd' )
		{ 
			int leftKey,rightKey;
			scanf(" %d,%d", &leftKey,&rightKey);
			delete_interval( searchtree,leftKey,rightKey);
		}
		else if (nextop == 'p')
		{
			printf("\nPrinting Tree - \n");
			printTree(searchtree,0);
			//printTreeList(searchtree);
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

void delete_interval(m_tree_t * tree, int leftKey, int rightKey)
{
	_delete_balanced(tree,leftKey,leftKey,rightKey);
	_delete_balanced(tree,rightKey,leftKey,rightKey);
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

		fixList(root->key,root->list,root->right->list);
		fixList(root->key,root->list,root->left->list);
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

interval_node * addToList(interval_node *list, unsigned int leftInterval,unsigned int rightInterval)
{
	interval_node *new_node;
	new_node = (interval_node *) malloc(sizeof(interval_node));
	new_node->leftInterval = leftInterval;
	new_node->rightInterval = rightInterval;
	new_node->next = NULL;
	if (list == NULL)
	{
		list = new_node;
	}
	else
	{
		list->next = new_node;
	}
	return list;
}

interval_node *deleteFromList(interval_node *list, unsigned int leftInterval,unsigned int rightInterval)
{
	interval_node *temp,*prev;
	temp = prev = list;

	while (temp != NULL)
	{
		if (temp->leftInterval == leftInterval && temp->rightInterval == rightInterval)
		{
			if (temp == list)
			{
				list = temp->next;
				free(temp);
				//printf("\nList after deleting (head case) - \n");
				//printList(list);
				return list;
			}
			else
			{
				prev->next = temp->next;
				free(temp);
				temp = NULL;
				//printf("\nList after deleting - \n");
				//printList(list);
				return list;
			}
		}
		else
		{
			prev = temp;
			temp = temp->next;
		}
	}
	return list;
}

void freeList(interval_node *list)
{
	interval_node *temp,*prev;
	temp = prev = list;
	if (temp != NULL)
	{
		if (temp->next == NULL)
		{
			free(temp);
			temp = NULL;
			return;
		}
		prev = temp;
		temp = temp->next;

		while (temp != NULL)
		{
			free(prev);
			prev = NULL;
			prev = temp;
			temp = temp->next;
		}
	}
}

int intervalInList(interval_node *list,unsigned int leftInterval,unsigned int rightInterval)
{
	interval_node *temp = NULL;
	temp = list;
	while (temp != NULL)
	{
		if (temp->leftInterval == leftInterval && temp->rightInterval == rightInterval)
		{
			return 1;
		}
		temp = temp->next;
	}
	return 0;
}

unsigned int getSmallest(interval_node *list)
{
	unsigned int min = 32767;
	interval_node *temp;
	temp = list;
	while (temp != NULL)
	{
		if (temp->leftInterval < min)
		{
			min = temp->leftInterval;
		}
		temp = temp->next;
	}
	return min;
}

unsigned int getLargest(interval_node *list)
{
	unsigned int max = 0;
	interval_node *temp;
	temp = list;
	while (temp != NULL)
	{
		if (temp->rightInterval > max)
		{
			max = temp->rightInterval;
		}
		temp = temp->next;
	}
	return max;
}

void changeIntervals(m_tree_t *node,unsigned int leftInterval,unsigned int rightInterval)
{
	if (node != NULL)
	{
		if (node->leftInterval == leftInterval)
		{
			node->leftInterval = getSmallest(node->list);
		}
		if (node->rightInterval == rightInterval)
		{
			node->rightInterval = getLargest(node->list);
		}

		if (node->leftmin == leftInterval)
		{
			node->leftmin = getSmallest(node->list);
		}
		if (node->rightmax == rightInterval)
		{
			node->rightmax = getLargest(node->list);
		}
	}
}

int isInList(interval_node *list,unsigned int key)
{
	interval_node *temp;
	temp = list;
	while (temp != NULL)
	{
		if (temp->leftInterval == key)
		{
			return 1;
		}
		temp = temp->next;
	}
	return 0;
}

void printTreeList(m_tree_t *node)
{
	if (node->height > 0)
	{
		printTreeList(node->left);
		printTreeList(node->right);
		printf("\n (%p) %d: ",(void *) &node->list ,node->key);
		printList(node->list);
	}
	else
	{
		printf("\n (%p) %d: ",(void *) &node->list ,node->key);
		printList(node->list);
	}
}
void printList(interval_node *node)
{
	interval_node *temp;
	temp = node;
	while (temp != NULL)
	{
		printf(" %d,%d ",temp->leftInterval,temp->rightInterval);
		temp = temp->next;
	}
	printf(" END");
}

void fixList(unsigned int key,interval_node *this_list,interval_node *other_list)
{
	interval_node *temp;
	unsigned int l,r;
	temp = other_list;
	while (temp != NULL)
	{
		l = temp->leftInterval;
		r = temp->rightInterval;
		if (key >= l && key < r )
		{
			if (!intervalInList(this_list,temp->leftInterval,temp->rightInterval))
			{
				this_list = addToList(this_list,temp->leftInterval,temp->rightInterval);
			}
		}
		temp = temp->next;
	}

	temp = this_list;
	while (temp != NULL)
	{
		l = temp->leftInterval;
		r = temp->rightInterval;
		if (key < l || key > r)
		{
			temp = temp->next;
			deleteFromList(this_list,l,r);
		}
		else
		{
			temp = temp->next;
		}
	}
}

interval_node *cloneList(interval_node *list)
{
	interval_node *new_list;
	new_list = (interval_node *) malloc(sizeof(interval_node));
	new_list = appendList(new_list,list);
	return new_list;
}


interval_node *appendList(interval_node *list,interval_node *toAppend)
{
	interval_node *temp;
	temp = toAppend;
	while (temp != NULL)
	{
		list = addToList(list,temp->leftInterval,temp->rightInterval);
		temp = temp->next;
	}
	return list;
}
