#include <cstdio>
#include <cstdlib>
#include <map>


#define BLOCKSIZE 256

typedef int key_t;
typedef int object_t;

#define BEFORE 1
#define AFTER 0

//#define STEP

typedef struct tr_n_t 
{
	key_t      key; 
	struct tr_n_t  *left;
	struct tr_n_t  *right;
	struct tr_n_t  *parent;
	/* possibly additional information */ 
	unsigned int height;
} o_t;

typedef std::map<int,o_t *> address_map_t;
int top, bottom;
//int prev_global_maximum, prev_global_minimum;

address_map_t address_map;

//Forward Declarations
void indent(char ch, int level);
void printTree (o_t *root, int level);
void printMap();
void updateAddressMap(o_t *node);
void deleteFromMap(int delete_key);

void insert_before(o_t *ord, key_t a, key_t b);
void insert_after(o_t *ord, key_t a, key_t b);
void insert_top(o_t *ord, key_t a);
void insert_bottom(o_t *ord, key_t a);
void delete_o(o_t *ord, key_t a);
int is_before(o_t *ord, key_t a, key_t b);

void right_rotation(o_t *n)
{
	o_t *tmp_node;
	key_t tmp_key;
	tmp_node = n->right;
	tmp_key = n->key;

	n->right = n->left;
	n->key = n->left->key;

	n->left = n->right->left;
	n->right->left = n->right->right;

	n->right->right = tmp_node;
	n->right->key = tmp_key;

	n->left->parent = n->right->parent = n;
	n->right->right->parent = n->right->left->parent = n->right;
	n->left->left->parent = n->left->right->parent = n->left;
}

void left_rotation(o_t *n)
{
	o_t *tmp_node;
	key_t tmp_key;
	tmp_node = n->left;
	tmp_key = n->key;
	n->left = n->right;
	n->key = n->right->key;
	n->right = n->left->right;
	n->left->right = n->left->left;
	n->left->left = tmp_node;
	n->left->key = tmp_key;


	n->left->parent = n->right->parent = n;
	n->left->left->parent = n->left->right->parent = n->left;
	n->right->right->parent = n->right->left->parent = n->right;
}

o_t *currentblock = NULL;
int    size_left;
o_t *free_list = NULL;
int nodes_taken = 0;
int nodes_returned = 0;

void checkOrSetGlobalMinMax(int new_key);

o_t *get_node()
{ 
	o_t *tmp;
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
			currentblock = (o_t *) malloc( BLOCKSIZE * sizeof(o_t) );
			size_left = BLOCKSIZE;
		}
		tmp = currentblock++;
		size_left -= 1;
	}
	tmp->left = tmp->right = tmp->parent = NULL;
	tmp->height = 0;
	return( tmp );
}

void return_node(o_t *node)
{  
	node->right = free_list;
	free_list = node;
	nodes_returned +=1;
}

o_t *create_order(void)
{
	o_t *tmp_node;
	tmp_node = get_node();
	tmp_node->left = NULL;
	top = bottom = 0;
	return( tmp_node );
}

void delete_balanced(o_t *tree, key_t delete_key)
{
	o_t *tmp_node, *upper_node, *other_node;
	int finished;

	object_t *deleted_object;
	if( tree->left == NULL )
		return ;
	else if( tree->right == NULL && tree->parent == NULL)
	{
		if( tree->key == delete_key )
		{
			deleted_object = (object_t *) tree->left;
			tree->left = NULL;
			deleteFromMap(delete_key);
		}
		else
			return;
	}
	else
	{
		tmp_node = tree;
		if( tmp_node->key != delete_key )
			return;
		else
		{
			upper_node = tmp_node->parent;

			if (tmp_node == upper_node->left)
				other_node = upper_node->right;
			else
				other_node = upper_node->left;

			upper_node->key   = other_node->key;
			upper_node->left  = other_node->left;
			upper_node->right = other_node->right;

			if (upper_node->height > 1)
			{
				upper_node->left->parent = upper_node->right->parent = upper_node;
			}

			updateAddressMap(upper_node);
			deleteFromMap(delete_key);
			if (top == delete_key)
			{
				top = upper_node->key;
			}
			else if (bottom == delete_key)
			{
				bottom = upper_node->key;
			}

			deleted_object = (object_t *) tmp_node->left;
			return_node( tmp_node );
			return_node( other_node );
			//return( deleted_object );
		}
		tmp_node = upper_node;
		/* rebalance */
		if (tmp_node->parent == NULL)
		{
			finished = 1;
			tmp_node->height = 0;
		}
		else
		{
			finished = 0;
		}
		while( tmp_node != NULL && !finished )
		{
			int tmp_height, old_height;
			old_height = tmp_node->height;

			if (tmp_node->right == NULL)
			{
				tmp_node->height = 0;
				tmp_node = tmp_node->parent;
				continue;
			}
			else if( tmp_node->left->height - tmp_node->right->height == 2 )
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
			if( tmp_node->height == old_height )
				finished = 1;
			tmp_node = tmp_node->parent;
		}
	}
}

void remove_tree(o_t *tree)
{
	o_t *current_node, *tmp;
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

int insert_balanced(o_t *tree, key_t new_key, int beforeOrAfter)
{
	o_t *tmp_node;
	int finished;
	if( tree->left == NULL )
	{
		tree->left = (o_t *) 0xbadbeef;
		tree->key = new_key;
		tree->height = 0;
		tree->right = NULL;
		tree->parent = NULL;
		//checkOrSetGlobalMinMax(new_key);
		//top = new_key;
		//bottom = new_key;
		updateAddressMap(tree);
	}
	else
	{
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			if( new_key < tmp_node->key )
				tmp_node = tmp_node->left;
			else
				tmp_node = tmp_node->right;
		}
		if( tmp_node->key == new_key )
			return( -1 );
		/* key is distinct, now perform the insert */ 
		else
		{

			o_t *old_leaf, *new_leaf;
			old_leaf = get_node();
			old_leaf->left = tmp_node->left;
			old_leaf->key = tmp_node->key;
			old_leaf->right= NULL;
			old_leaf->height = 0;
			new_leaf = get_node();
			new_leaf->left = (o_t *) 0xbadbeef;
			new_leaf->key = new_key;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			if( beforeOrAfter == AFTER)
			{
				tmp_node->left = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = new_key;
			}
			else if (beforeOrAfter == BEFORE)
			{
				tmp_node->left = new_leaf;
				tmp_node->right = old_leaf;
			}
			tmp_node->height = 1;
			old_leaf->parent = new_leaf->parent = tmp_node;
			//checkOrSetGlobalMinMax(new_key);
			updateAddressMap(new_leaf);
			updateAddressMap(old_leaf);
			tmp_node = tmp_node->parent;
		}
		/* rebalance */
		finished = 0;
		while( tmp_node != NULL && !finished)
		{
			int tmp_height, old_height;

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
			else /* update height even if there
				 was no rotation */
			{
				if( tmp_node->left->height > tmp_node->right->height )
					tmp_node->height = tmp_node->left->height + 1;
				else
					tmp_node->height = tmp_node->right->height + 1;
			}
			if( tmp_node->height == old_height )
				finished = 1;
			tmp_node = tmp_node->parent;
		}
	}
	return( 0 );
}

#ifdef STEP

int main()
{
	o_t *searchtree;
	char nextop;
	searchtree = create_order();

	while ((nextop = getchar()) != 'q')
	{ 
		if ( nextop == 't' )
		{
			int inskey;
			scanf(" %d", &inskey);
			printf("insert_top: %d\n",inskey);
			insert_top(searchtree,inskey);
		}
		else if ( nextop == 'b' )
		{
			int inskey;
			scanf(" %d", &inskey);
			printf("insert_bottom: %d\n",inskey);
			insert_bottom(searchtree,inskey);
		}
		else if ( nextop == 'p' )
		{
			printf("\nTop/Bottom - (%d,%d).\n",top,bottom);
			printTree(searchtree,0);
			printMap();
		}
		else if ( nextop == 'i' )
		{
			int a,b,is_before_val;
			scanf(" %d %d", &a,&b);
			is_before_val = is_before(searchtree,a,b);
			if (is_before_val == 1)
			{
				printf("%d is before %d\n",a,b);
			}
			else 
			{
				printf("%d is not before %d\n",a,b);
			}
		}
		else if ( nextop == 'e' )
		{
			int a,b;
			scanf(" %d %d", &a,&b);
			printf("Inserting %d before %d\n",a,b);
			insert_before(searchtree,a,b);
		}
		else if ( nextop == 'a' )
		{
			int a,b;
			scanf(" %d %d", &a,&b);
			printf("Inserting %d after %d\n",a,b);
			insert_after(searchtree,a,b);
		}
		else if ( nextop == 'd' )
		{ 
			int delkey, *delobj;
			scanf(" %d", &delkey);
			delobj = delete_balanced( searchtree, delkey);
			if( delobj == NULL )
				printf("  delete failed for key %d\n", delkey);
			else
				printf("  delete successful, deleted object %d for key %d\n", *delobj, delkey);
		}
		else if ( nextop == 'r' )
		{
			int a;
			scanf(" %d", &a);
			printf("Deleted %d\n",a);
			delete_o(searchtree,a);
		}
	}

	remove_tree( searchtree );
	printf("Removed tree.\n");
	printf("Total number of nodes taken %d, total number of nodes returned %d\n",
		nodes_taken, nodes_returned );
	return(0);
}

#endif

void indent(char ch, int level)
{
	int i;
	for (i = 0; i < level; i++ )
	putchar(ch);
}

void printTree (o_t *root, int level)
{
	if (root->right == NULL)
	{
		indent('\t', level);
		printf("[%lu] %d (%d)\n", (unsigned long) root, root->key,level);
	}
	else
	{
		printTree(root->right, level + 1);
		indent('\t', level);
		printf("[%lu] %d (%d)\n", (unsigned long) root, root->key,level);
		printTree(root->left, level + 1);
	}
}

/*void checkOrSetGlobalMinMax(int new_key)
{
	if (new_key < global_minimum)
	{
		prev_global_minimum = global_minimum;
		global_minimum = new_key;
	}
	if (new_key > global_maximum)
	{
		prev_global_maximum = global_maximum;
		global_maximum = new_key;
	}
}*/

void updateAddressMap(o_t *node)
{
	address_map[node->key] = node;
}

void printMap()
{
	address_map_t::iterator it;
	printf("\n\nAddress map contents - ");
	for (it = address_map.begin(); it != address_map.end(); it++)
	{
		printf("\n%d - %lu",(int) it->first, (unsigned long) it->second);
	}
	return;
}

void insert_before(o_t *ord, key_t a, key_t b)
{
	o_t *node_b;
	node_b = address_map[b];
	if (node_b != NULL)
	{
		insert_balanced(node_b,a,BEFORE);
		if (b == bottom)
		{
			bottom = a;
		}
	}
	else
	{
		printf("\nNode %d not found in the address Map!",b);
	}
}

void insert_after(o_t *ord, key_t a, key_t b)
{
	o_t *node_b;
	node_b = address_map[b];
	if (node_b != NULL)
	{
		insert_balanced(node_b,a,AFTER);
		if (b == top)
		{
			top = a;
		}
	}
	else
	{
		printf("\nNode %d not found in the address Map!",b);
	}
}

void insert_top(o_t *ord, key_t a)
{
	o_t *node_b;
	if (ord->left != NULL)
	{
		node_b = address_map[top];
	}
	else
	{
		node_b = ord;
		bottom = a;
	}
	if (node_b != NULL)
	{
		insert_balanced(node_b,a,AFTER);
		top = a;
	}
	else
	{
		printf("\nNode %d not found in the address Map!",top);
	}
}

void insert_bottom(o_t *ord, key_t a)
{
	o_t *node_b;
	if (ord->left != NULL)
	{
		node_b = address_map[bottom];
	}
	else
	{
		node_b = ord;
		top = a;
	}
	if (node_b != NULL)
	{
		insert_balanced(node_b,a,BEFORE);
		bottom = a;
	}
	else
	{
		printf("\nNode %d not found in the address Map!",bottom);
	}
}

int is_before(o_t *ord, key_t a, key_t b)
{
	o_t *node_a, *node_b;
	unsigned long traversalArray[32], i = 0;
	if (ord->left != NULL)
	{
		node_a = address_map[a];
		node_b = address_map[b];
		if (node_a == NULL)
		{
			printf("\nNode %d not found in the address Map!",a);
		}
		if (node_b == NULL)
		{
			printf("\nNode %d not found in the address Map!",b);
		}
		for (i = 0 ;i < 32; i++)
		{
			traversalArray[i] = 0;
		}
		i = 0;
		while (node_a != NULL)
		{
			traversalArray[i] = (unsigned long) node_a;
			node_a = node_a->parent;
			i++;
		}
		while (node_b != ord)
		{
			i = 0;
			while ( traversalArray[i] != 0)
			{
				if (traversalArray[i] == (unsigned long) node_b->parent)
				{
					if (node_b == node_b->parent->left)
					{
						return 0;
					}
					else if (node_b == node_b->parent->right)
					{
						return 1;
					}
				}
				else
				{
					i++;
				}
			}
			node_b = node_b->parent;
		}
	}
	return 0;
}

void delete_o(o_t *ord, key_t a)
{
	o_t *node_a;
	node_a = address_map[a];
	delete_balanced(node_a,a);
}

void deleteFromMap(int delete_key)
{
	if (address_map[delete_key] != NULL)
	{
		address_map.erase(delete_key);
	}
	else
	{
		printf("\n %d not in address map!",delete_key);
	}
}
long p(long q)
{
	return( (1247*q +104729) % 300007 );
}

#ifndef STEP
int main()
{
	long i; o_t *o; 
	printf("starting \n");
	o = create_order();
	for(i=100000; i>=0; i-- )
		insert_bottom( o, p(i) );
	for(i=100001; i< 300007; i+=2 )
	{
		insert_after(o, p(i+1), p(i-1) );
		insert_before( o, p(i), p(i+1) );
	}
	printf("inserted 300000 elements. ");
	for(i = 250000; i < 300007; i++ )
		delete_o( o, p(i) );
	printf("deleted 50000 elements. ");
	insert_top( o, p(300006) );
	for(i = 250000; i < 300006; i++)
		insert_before( o, p(i) , p(300006) );
	printf("reinserted. now testing order\n");
	for( i=0; i < 299000; i +=42 )
	{
		if( is_before( o, p(i), p(i+23) ) != 1 )
		{
			printf(" found error (1) \n"); exit(0);
		}
	}
	for( i=300006; i >57; i -=119 )
	{
		if( is_before( o, p(i), p(i-57) ) != 0 )
		{
			printf(" found error (0) \n"); exit(0);
		}
	}
	printf("finished. no problem found.\n");
}
#endif

