#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define BLOCKSIZE 256

typedef int key_t;
typedef int object_t;
typedef struct tr_n_t 
{
	key_t key; 
	struct tr_n_t *left;
	struct tr_n_t *right;
	unsigned int height, left_leaves, right_leaves;
} tree_node_t;

typedef struct st_t
{
	tree_node_t *item;
	struct st_t *next;
} stack_t;

typedef struct lst_t
{
	char *data;
	struct lst_t *next;
} list_t;

typedef struct txt_t
{
	tree_node_t *searchtree;
	unsigned int txt_length;
} text_t;

int fix_child_key(tree_node_t *parent_node,tree_node_t *child_node)
{
	int new_child_key;
	if (parent_node->left == child_node)
	{
		if (child_node->right == NULL)
		{
			new_child_key = parent_node->key - 1;
		}
		else
		{
			new_child_key = parent_node->key - child_node->right_leaves;
		}
	}
	else if (parent_node->right == child_node)
	{
		if (child_node->right == NULL)
		{
			new_child_key = parent_node->key;
		}
		else
		{
			new_child_key = parent_node->key + child_node->left_leaves;
		}
	}
	else
	{
		printf("\nError! bad node sent to fix_child_key");
	}
	return new_child_key;
}


int fix_parent_key(tree_node_t *parent_node,tree_node_t *child_node)
{
	int new_parent_key;
	if (parent_node->left == child_node)
	{
		new_parent_key = child_node->key + child_node->right_leaves;
	}
	else if (parent_node->right == child_node)
	{
		new_parent_key = child_node->key - child_node->left_leaves;
	}
	else
	{
		printf("\nError! bad node sent to fix_parent_key");
	}
	return new_parent_key;
}

void update_leafcount(tree_node_t *tmp_node)
{
	int done = 0;
	if (tmp_node->left != NULL)
	{
		if (tmp_node->left->right == NULL)
		{
			tmp_node->left_leaves = 1;
		}
		else
		{
			tmp_node->left_leaves = tmp_node->left->left_leaves + tmp_node->left->right_leaves;
		}
	}
	else
	{
		tmp_node->left_leaves = 0;
	}
	if (tmp_node->right != NULL)
	{
		if (tmp_node->right->right == NULL)
		{
			tmp_node->right_leaves = 1;
		}
		else
		{
			tmp_node->right_leaves = tmp_node->right->right_leaves + tmp_node->right->left_leaves;
		}
	}
	else
	{
		tmp_node->right_leaves = 0;
	}
}

//Forward Declarations
stack_t *get_node_st();
void return_node_st(stack_t *node);
void remove_tree(tree_node_t *tree);
int insert_balanced(tree_node_t *tree, key_t new_key,object_t *new_object);

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

void push( tree_node_t *x, stack_t *st)
{
	stack_t *tmp;
	tmp = get_node_st();
	tmp->item = x;
	tmp->next = st->next;
	st->next = tmp;
}

tree_node_t *pop(stack_t *st)
{
	stack_t *tmp;
	tree_node_t *tmp_item;
	tmp = st->next;
	if (tmp != NULL)
	{
		st->next = tmp->next;
		tmp_item = tmp->item;
		return_node_st( tmp );
		return( tmp_item );
	}
	else
	{
		return NULL;
	}
}

tree_node_t *top_element(stack_t *st)
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

void right_rotation(tree_node_t *n)
{
	tree_node_t *tmp_node;
	key_t tmp_key;
	int changed;
	tmp_node = n->right;
	tmp_key = n->key;
	changed = n->left->right->key;
	n->right = n->left;
	n->key = n->left->key;
	n->left = n->right->left;
	n->right->left = n->right->right;
	n->right->right = tmp_node;
	n->right->key = tmp_key - n->key + changed;
	n->left->key = n->key - changed;
	n->key = tmp_key;
}

void left_rotation(tree_node_t *n)
{
	tree_node_t *tmp_node;
	key_t tmp_key;
	int changed;
	tmp_node = n->left;
	tmp_key = n->key;
	changed = n->right->left->key;
	n->left = n->right;
	n->key = n->right->key;
	n->right = n->left->right;
	n->left->right = n->left->left;
	n->left->left = tmp_node;
	n->left->key = tmp_key - n->key + changed;
	n->right->key = n->key - changed;
	n->key = tmp_key;

}

void indent(char ch, int level)
{
	int i;
	for (i = 0; i < level; i++ )
	putchar(ch);
}

void printTree (tree_node_t *root, int level)
{
	if (root->right == NULL)
	{
		indent('\t', level);
		printf("%d (%d) (%d,%d)\n", root->key,level,root->left_leaves,root->right_leaves);
	}
	else
	{
		printTree(root->right, level + 1);
		indent('\t', level);
		printf("%d (%d) (%d,%d)\n", root->key,level,root->left_leaves,root->right_leaves);
		printTree(root->left, level + 1);
	}
}


tree_node_t *currentblock = NULL;
int    size_left;
tree_node_t *free_list = NULL;
int nodes_taken = 0;
int nodes_returned = 0;

stack_t *currentblock_st = NULL;
int size_left_st;
stack_t *free_list_st = NULL;
int nodes_taken_st = 0;
int nodes_returned_st = 0;

tree_node_t *get_node()
{ 
	tree_node_t *tmp;
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
			currentblock = (tree_node_t *) malloc( BLOCKSIZE * sizeof(tree_node_t) );
			size_left = BLOCKSIZE;
		}
		tmp = currentblock++;
		size_left -= 1;
	}
	//tmp->left_leaves = tmp->right_leaves = 0;
	return( tmp );
}

void return_node(tree_node_t *node)
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

list_t *currentblock_list = NULL;
int size_left_list;
list_t *free_list_list = NULL;
int nodes_taken_list = 0;
int nodes_returned_list = 0;

#define BLOCKSIZE_STRING 128

list_t *get_string()
{ 
	list_t *tmp;
	tmp = (list_t *)malloc(sizeof(list_t));
	return(tmp);
}

void return_string(list_t *node)
{
	free(node);
}

tree_node_t *create_tree(void)
{
	tree_node_t *tmp_node;
	tmp_node = get_node();
	tmp_node->left = NULL;
	return( tmp_node );
}

object_t *find_iterative(tree_node_t *tree, key_t query_key)
{ 
	tree_node_t *tmp_node;
	if( tree->left == NULL )
		return(NULL);
	else
	{
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			if( query_key < tmp_node->key )
				tmp_node = tmp_node->left;
			else
				tmp_node = tmp_node->right;
		}
		if( tmp_node->key == query_key )
			return( (object_t *) tmp_node->left );
		else
			return( NULL );
	}
}

object_t *find_recursive(tree_node_t *tree, key_t query_key)
{  
	if( tree->left == NULL || (tree->right == NULL && tree->key != query_key ) )
		return(NULL);
	else if (tree->right == NULL && tree->key == query_key )
		return( (object_t *) tree->left );     
	else
	{  
		if( query_key < tree->key )
			return( find_recursive(tree->left, query_key) );
		else
			return( find_recursive(tree->right, query_key) );
	}
}

int insert(tree_node_t *tree, key_t new_key, object_t *new_object)
{
	tree_node_t *tmp_node;
	if( tree->left == NULL )
	{
		tree->left = (tree_node_t *) new_object;
		tree->key  = new_key;
		tree->right  = NULL; 
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
		/* found the candidate leaf. Test whether key distinct */
		if( tmp_node->key == new_key )
			return( -1 );
		/* key is distinct, now perform the insert */ 
		else
		{
			tree_node_t *old_leaf, *new_leaf;
			old_leaf = get_node();
			old_leaf->left = tmp_node->left; 
			old_leaf->key = tmp_node->key;
			old_leaf->right  = NULL;
			new_leaf = get_node();
			new_leaf->left = (tree_node_t *) new_object; 
			new_leaf->key = new_key;
			new_leaf->right  = NULL; 
			if( tmp_node->key < new_key )
			{
				tmp_node->left  = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = new_key;
			}
			else
			{
				tmp_node->left  = new_leaf;
				tmp_node->right = old_leaf;
			}
		}
	}
	return( 0 );
}

object_t *_delete(tree_node_t *tree, key_t delete_key)
{
	tree_node_t *tmp_node, *upper_node, *other_node;
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
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
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
	}
}

text_t *create_text()
{
	text_t *text;
	text = (text_t *) malloc(sizeof(text_t));
	text->searchtree = create_tree();
	return text;
}

void remove_text(text_t *text)
{
	remove_tree(text->searchtree);
	free(text);
}

int length_text(text_t *txt)
{
	if (txt->searchtree->left == NULL)
	{
		return 0;
	}
	else
		return (txt->searchtree->key);
}

int actual_index(text_t *txt,int line_num)
{
	tree_node_t *tmp_node;
	tmp_node = txt->searchtree;

	if( tmp_node->left == NULL )
		return(NULL);
	else
	{
		while( tmp_node->right != NULL )
		{
			if ( line_num > (tmp_node->key - tmp_node->left_leaves) &&
					line_num < (tmp_node->key - 1) )
				tmp_node = tmp_node->left;
			else
				tmp_node = tmp_node->right;
		}
		if( tmp_node->key == line_num )
			//return ( ((list_t*) tmp_node->left)->data);
				return 0;
	}
}

char *get_line(text_t *txt,unsigned int index)
{
	tree_node_t *tmp_node,*last_node;
	tmp_node = txt->searchtree;

	if( tmp_node->left == NULL )
		return(NULL);
	else
	{
		while( tmp_node->right != NULL )
		{

			if ( index <= tmp_node->left->key )
			{
				tmp_node = tmp_node->left;
			}
			else
			{
				index = index - tmp_node->left->key;
				tmp_node = tmp_node->right;
			}
		}
		if( tmp_node->key == index )
			return ( ((list_t *) tmp_node->left)->data);
		else
			return( NULL );
	}
}
void insert_line(text_t *txt,int index,char *new_line)
{
	int retCode;
	list_t *string_element;
	string_element = get_string();
	string_element->data = strdup(new_line);
	
	if ( (retCode = insert_balanced(txt->searchtree,(key_t) index,(object_t *) string_element) != 0))
	{
		printf("\nERROR: insert_balanced failed. RetCode - %d",retCode);
	}
}

void append_line(text_t *txt,char *new_line)
{
	tree_node_t *root_node;
	root_node = txt->searchtree;
	if (root_node->left == NULL)
	{
		list_t *string_element;
		string_element = get_string();
		string_element->data = strdup(new_line);
		root_node->key = 1;
		root_node->left = (tree_node_t *) string_element;
		root_node->right = NULL;
		root_node->height = 0;
	}
	else
	{
		insert_line(txt,length_text(txt) + 1,new_line);
	}
}

char *set_line(text_t *txt,int index,char *new_line)
{
	tree_node_t *tree;
	tree = txt->searchtree;
	if( tree->left == NULL )
	{
		return NULL;
	}
	else
	{
		tree_node_t *tmp_node;
		tmp_node = tree;
		while( tmp_node->right != NULL )
		{
			if( index <= tmp_node->left->key )
				tmp_node = tmp_node->left;
			else
			{
				index = index - tmp_node->left->key;
				tmp_node = tmp_node->right;
			}
			
		}
		if( tmp_node->key == index )
		{
			char *line;
			line = ((list_t *) (tmp_node->left))->data;
			//free(line);
			((list_t *) (tmp_node->left))->data = strdup(new_line);
			return line;
		}
		else
		{
			return NULL;
		}
	}
}
object_t *_delete_balanced(tree_node_t *tree, key_t delete_key);
char *delete_line(text_t *txt,int index)
{
	if(txt == NULL)
		return NULL;
	if(txt->searchtree == NULL)
		return NULL;
	return (char *)((list_t *)_delete_balanced(txt->searchtree,(key_t) index))->data;
}

object_t *_delete_balanced(tree_node_t *tree, key_t delete_key)
{
	tree_node_t *tmp_node, *upper_node, *other_node;
	//int finished;
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
			if( delete_key <= tmp_node->left->key )
			{  
				tmp_node   = upper_node->left; 
				other_node = upper_node->right;
			} 
			else
			{
				delete_key = delete_key  - tmp_node->left->key;
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
			upper_node->height = 0;
			deleted_object = (object_t *) tmp_node->left;
			return_node( tmp_node );
			return_node( other_node );
			update_leafcount(upper_node);
		}
		/* rebalance */
		//finished = 0;
		/*if (!stack_empty(stack))
		{
			print_stack(stack);
		}*/
		//Throw out the top of stack.
		pop(stack);
		while( !stack_empty(stack))// && !finished )
		{
			int tmp_height, old_height;
			tmp_node = pop(stack);
			tmp_node->key = tmp_node->left->key + tmp_node->right->key;
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
			update_leafcount(tmp_node);
			/*if( tmp_node->height == old_height )
				finished = 1;*/
		}
		remove_stack(stack);
	}
	return( deleted_object );
}

void remove_tree(tree_node_t *tree)
{
	tree_node_t *current_node, *tmp;
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

tree_node_t *interval_find(tree_node_t *tree, key_t a, key_t b)
{ 
	tree_node_t *tr_node;
	tree_node_t *node_stack[200]; int stack_p = 0;
	tree_node_t *result_list, *tmp;
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

void check_tree( tree_node_t *tr, int depth, int lower, int upper )
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
			printf("%d (%d) (%d,%d)  ", tr->key, depth,tr->left_leaves,tr->right_leaves );
		else
			printf("Wrong Object \n");
	}
	else
	{
		check_tree(tr->left, depth+1, lower, tr->key ); 
		check_tree(tr->right, depth+1, tr->key, upper ); 
	}
}


int insert_balanced(tree_node_t *tree, key_t new_key,object_t *new_object)
{
	tree_node_t *tmp_node;
	//int finished;
	stack_t *stack;
	int inserted_key = 32767;
	tree_node_t *last_node;
	if( tree->left == NULL )
	{
		tree->left = (tree_node_t *) new_object;
		tree->key = new_key;
		tree->height = 0;
		tree->left_leaves = tree->right_leaves = 0;
		tree->right = NULL;
	}
	else
	{
		stack = create_stack();
		tmp_node = tree;
		last_node = tmp_node;
		while( tmp_node->right != NULL )
		{
			push(tmp_node,stack);
			last_node = tmp_node;
			if( new_key <= tmp_node->left->key )
				tmp_node = tmp_node->left;
			else
			{
				new_key = new_key - tmp_node->left->key;
				tmp_node = tmp_node->right;
			}
			
		}
		/* found the candidate leaf. Test whether key distinct */
		if( tmp_node->key == new_key )
		{
			tree_node_t *old_leaf, *new_leaf;
			old_leaf = get_node();
			old_leaf->left = tmp_node->left;
			old_leaf->key = tmp_node->key;
			old_leaf->right= NULL;
			old_leaf->height = 0;

			new_leaf = get_node();
			new_leaf->left = (tree_node_t *) new_object;
			new_leaf->key = 1;
			new_leaf->right = NULL;
			new_leaf->height = 0;

			tmp_node->left = new_leaf;
			tmp_node->right = old_leaf;
			tmp_node->height = 1;
			tmp_node->key = new_leaf->key + old_leaf->key;

		}
		else 
		{
			tree_node_t *old_leaf, *new_leaf;
			old_leaf = get_node();
			old_leaf->left = tmp_node->left;
			old_leaf->key = tmp_node->key;
			old_leaf->right= NULL;
			old_leaf->height = 0;

			new_leaf = get_node();
			new_leaf->left = (tree_node_t *) new_object;
			new_leaf->key = 1;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			tmp_node->left = old_leaf;
			tmp_node->right = new_leaf;
			tmp_node->key = new_leaf->key + old_leaf->key;
			tmp_node->height = 1;
		}
		/* rebalance */

		while( !stack_empty(stack))// && !finished )
		{
			int tmp_height, old_height;
			tmp_node = pop(stack);
			tmp_node->key = tmp_node->left->key + tmp_node->right->key;
			old_height= tmp_node->height;
			
			if( tmp_node->left->height - tmp_node->right->height == 2 )
			{
				if( tmp_node->left->left->height - tmp_node->right->height == 1 )
				{
					right_rotation( tmp_node );
					tmp_node->right->height = tmp_node->right->left->height + 1;
					tmp_node->height = tmp_node->right->height + 1;
				}
				else
				{
					left_rotation(tmp_node->left);
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
			else /* update height even if there was no rotation */
			{
				if( tmp_node->left->height > tmp_node->right->height )
				{
					tmp_node->height = tmp_node->left->height + 1;
				}
				else
				{
					tmp_node->height = tmp_node->right->height + 1;
				}
			}

		}
		remove_stack(stack);
	}
	return( 0 );
}

/*int man()
{
	tree_node_t *searchtree;
	char nextop;
	searchtree = create_tree();
	printf("Made Tree\n");
	printf("In the following, the key n is associated wth the object 10n+2\n");
	while ((nextop = getchar()) != 'q')
	{ 
		if ( nextop == 'i' )
		{ 
			int inskey,*insobj, success;
			insobj = (int *) malloc(sizeof(int));
			scanf(" %d", &inskey);
			*insobj = 10*inskey+2;
			success = insert_balanced( searchtree, inskey, insobj );
			if ( success == 0 )
			{
				printf("  insert successful, key = %d, object value = %d, \n",
					inskey, *insobj);
			}
			else
				printf("  insert failed, success = %d\n", success);
		}
		else if ( nextop == 'f' )
		{ 
			int findkey, *findobj;
			scanf(" %d", &findkey);
			findobj = find_iterative( searchtree, findkey);
			if( findobj == NULL )
				printf("  find (iterative) failed, for key %d\n", findkey);
			else
				printf("  find (iterative) successful, found object %d\n", *findobj);
			findobj = find_recursive( searchtree, findkey);
			if( findobj == NULL )
				printf("  find (recursive) failed, for key %d\n", findkey);
			else
				printf("  find (recursive) successful, found object %d\n", *findobj);
		}
		else if ( nextop == 'v' )
		{ 
			int a, b;  tree_node_t *results, *tmp;
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
	printf("Total number of nodes taken %d, total number of nodes returned %d\n",
		nodes_taken_st, nodes_returned_st );
	return(0);
}*/

int main()
{  
	int i, tmp;
	text_t *txt1, *txt2;
	char *c;

	printf("starting \n");
	txt1 = create_text();
	txt2 = create_text();
	append_line(txt1, "line one" );
	if( (tmp = length_text(txt1)) != 1)
	{
		printf("Test 1: length should be 1, is %d\n", tmp);
		exit(-1);
	}
	append_line(txt1, "line hundred" );
	insert_line(txt1, 2, "line ninetynine" );
	insert_line(txt1, 2, "line ninetyeight" );
	insert_line(txt1, 2, "line ninetyseven" );
	insert_line(txt1, 2, "line ninetysix" );
	insert_line(txt1, 2, "line ninetyfive" );
	//printTree(txt1->searchtree,0);

	for( i = 2; i < 95; i++ )
	{
		insert_line(txt1, 2, "some filler line between 1 and 95" );
	}
	if ((tmp = length_text(txt1)) != 100)
	{
		printf("Test 2: length should be 100, is %d\n", tmp);
		exit(-1);
	}
	printf("found at line 1:   %s\n",get_line(txt1,  1));
	printf("found at line 2:   %s\n",get_line(txt1,  2));
	printf("found at line 99:  %s\n",get_line(txt1, 99));
	printf("found at line 100: %s\n",get_line(txt1,100));
	for(i=1; i<=10000; i++)
	{  
		if( i%2==1 )
			append_line(txt2, "A");
		else 
			append_line(txt2, "B");
	}
	if( (tmp = length_text(txt2)) != 10000)
	{  
		printf("Test 3: length should be 10000, is %d\n", tmp);
		exit(-1);
	}
	c = get_line(txt2, 9876 );
	if( *c != 'B')
	{  
		printf("Test 4: line 9876 of txt2 should be B, found %s\n", c);
		exit(-1);
	}
	for( i= 10000; i > 1; i-=2 )
	{  
		c = delete_line(txt2, i);
		if( *c != 'B')
		{  
			printf("Test 5: line %d of txt2 should be B, found %s\n", i, c);
			exit(-1);
		}
		append_line( txt2, c );
	}
	for( i=1; i<= 5000; i++ )
	{  
		c = get_line(txt2, i);
		if( *c != 'A')
		{  
			printf("Test 6: line %d of txt2 should be A, found %s\n", i, c);
			exit(-1);
		}
	}
	for( i=1; i<= 5000; i++ )
		delete_line(txt2, 1 );
	for( i=1; i<= 5000; i++ )
	{  
		c = get_line(txt2, i);
		if( *c != 'B')
		{  
			printf("Test 7: line %d of txt2 should be B, found %s\n", i, c); 
			exit(-1);
		}
	}
	set_line(txt1, 100, "the last line");
	for( i=99; i>=1; i-- )
		delete_line(txt1, i );
	printf("found at the last line:   %s\n",get_line(txt1,  1));
}

