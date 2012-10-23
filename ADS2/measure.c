#include <stdio.h>
#include <stdlib.h>

#define NEGATIVE_INFINITY -102400
#define INFINITY 1024000

#define BLOCKSIZE 256
//#define STEP

typedef int key_t;
typedef int object_t;

typedef struct interval_node_t
{
	int leftInterval, rightInterval;
	struct interval_node_t *next;
} interval_node;

typedef struct tr_n_t
{
	key_t key;
	struct tr_n_t *left;
	struct tr_n_t *right;
	int height;
	int leftmin, rightMax;
	int measure;
	int leftInterval, rightInterval;
	interval_node *list;
} m_tree_t;

//CHECK
m_tree_t *traversalHistory[512];

int numTraversals;
m_tree_t *currentblock = NULL;
int size_left;
m_tree_t *free_list = NULL;
int nodes_taken = 0;
int nodes_returned = 0;

int Max(int a, int b);
int Min(int a, int b);
int setRightMax(m_tree_t *node);
int setLeftMin(m_tree_t *node);
int setMeasure(m_tree_t *node);
int setLeafMeasure(m_tree_t *node);
int setNodeMeasure(m_tree_t *tempNode);
m_tree_t * get_node();

int insert_balanced(m_tree_t *tree, key_t new_key,object_t *new_object, int other_key, int direction);
object_t * delete_balanced(m_tree_t *tree, key_t delete_key, int otherkey, int direction);
void printTreeList(m_tree_t *node);
void printList(interval_node *node);
void insert_interval(m_tree_t *tree, int leftKey, int rightKey);


m_tree_t * create_m_tree()
{
	m_tree_t *tree;
	tree = get_node();
	tree->left = tree->right = NULL;
	tree->list = NULL;
	return (tree);
}

void insert_interval(m_tree_t *tree, int leftKey, int rightKey)
{
	int insobj;
	insobj = 10*leftKey+2;
	insert_balanced(tree, leftKey, &insobj,rightKey,1 );
	insobj = 10*rightKey+2;
	insert_balanced(tree, rightKey, &insobj,leftKey,2 );
	//recursiveFixxer(tree);
}

void delete_interval(m_tree_t * tree, int leftKey, int rightKey)
{
	if ( delete_balanced(tree,leftKey,rightKey,1) == NULL)
		printf("\nError deleting %d",leftKey);
	else
		delete_balanced(tree,rightKey,leftKey,2);
}

int query_length(m_tree_t *tree)
{
	return tree->measure;
}
void left_rotation(m_tree_t *n)
{
	m_tree_t *temp;
	key_t key;
	temp = n->left;
	key = n->key;

	n->left = n->right;
	n->key = n->right->key;
	n->right = n->left->right;
	n->left->right = n->left->left;

	n->left->left = temp;
	n->left->key = key;
}

void right_rotation(m_tree_t *n)
{
	m_tree_t *temp;
	key_t key;
	temp = n->right;
	key = n->key;

	n->right = n->left;
	n->key = n->left->key;
	n->left = n->right->left;
	n->right->left = n->right->right;

	n->right->right = temp;
	n->right->key = key;
}

//Integrate
int balanceTree()
{
	int finished = 0;
	int i;
	for (i = numTraversals - 1; ((i >= 0) && (!finished)); i--)
	{
		int tempHeight, old_height;
		m_tree_t* tempNode = traversalHistory[i];
		old_height = tempNode->height;
		if (tempNode->left->height - tempNode->right->height == 2)
		{
			if (tempNode->left->left->height - tempNode->right->height == 1)
			{
				right_rotation(tempNode);
				tempNode->right->leftInterval= tempNode->key;
				tempNode->right->rightInterval= tempNode->rightInterval;

				tempNode->right->leftmin = Min(tempNode->right->left->leftmin,tempNode->right->right->leftmin);
				tempNode->right->rightMax = Max(tempNode->right->left->rightMax,tempNode->right->right->rightMax);

				tempNode->leftmin = Min(tempNode->left->leftmin,tempNode->right->leftmin);
				tempNode->rightMax = Max(tempNode->left->rightMax,tempNode->right->rightMax);

				tempNode->right->height = tempNode->right->left->height + 1;
				tempNode->height = tempNode->right->height + 1;
				tempNode->right->measure = setNodeMeasure(tempNode->right);
				tempNode->measure = setNodeMeasure(tempNode);
			}
			else
			{
				left_rotation(tempNode->left);
				tempNode->left->left->rightInterval= tempNode->left->key;
				tempNode->left->left->leftInterval= tempNode->left->leftInterval;

				tempNode->left->left->leftmin = Min(tempNode->left->left->left->leftmin,tempNode->left->left->right->leftmin);
				tempNode->left->left->rightMax = Max(tempNode->left->left->left->rightMax,tempNode->left->left->right->rightMax);
				tempNode->left->leftmin = Min(tempNode->left->left->leftmin,tempNode->left->right->leftmin);
				tempNode->left->rightMax = Max(tempNode->left->left->rightMax,tempNode->left->right->rightMax);
				tempNode->left->left->measure = setNodeMeasure(tempNode->left->left);
				tempNode->left->measure = setNodeMeasure(tempNode->left);

				right_rotation(tempNode);
				tempNode->right->leftInterval= tempNode->key;
				tempNode->right->rightInterval= tempNode->rightInterval;
				tempNode->right->leftmin = Min(tempNode->right->left->leftmin,tempNode->right->right->leftmin);
				tempNode->right->rightMax = Max(tempNode->right->left->rightMax,tempNode->right->right->rightMax);
				tempNode->leftmin = Min(tempNode->left->leftmin,tempNode->right->leftmin);
				tempNode->rightMax = Max(tempNode->left->rightMax,tempNode->right->rightMax);
				tempNode->right->measure = setNodeMeasure(tempNode->right);
				tempNode->measure = setNodeMeasure(tempNode);

				tempHeight = tempNode->left->left->height;
				tempNode->left->height = tempHeight + 1;
				tempNode->right->height = tempHeight + 1;
				tempNode->height = tempHeight + 2;
			}
		}
		else if (tempNode->left->height - tempNode->right->height == -2)
		{
			if (tempNode->right->right->height - tempNode->left->height == 1)
			{
				left_rotation(tempNode);
				tempNode->left->rightInterval= tempNode->key;
				tempNode->left->leftInterval= tempNode->leftInterval;
				tempNode->left->leftmin = Min(tempNode->left->left->leftmin,tempNode->left->right->leftmin);
				tempNode->left->rightMax = Max(tempNode->left->left->rightMax,tempNode->left->right->rightMax);
				tempNode->leftmin = Min(tempNode->left->leftmin,tempNode->right->leftmin);
				tempNode->rightMax = Max(tempNode->left->rightMax,tempNode->right->rightMax);
				tempNode->left->measure = setNodeMeasure(tempNode->left);
				tempNode->measure = setNodeMeasure(tempNode);
				tempNode->left->height = tempNode->left->right->height + 1;
				tempNode->height = tempNode->left->height + 1;
			}
			else
			{
				right_rotation(tempNode->right);
				tempNode->right->right->leftInterval= tempNode->right->key;
				tempNode->right->right->rightInterval= tempNode->right->rightInterval;
				tempNode->right->right->leftmin = Min(tempNode->right->right->left->leftmin,tempNode->right->right->right->leftmin);
				tempNode->right->right->rightMax = Max(tempNode->right->right->left->rightMax,tempNode->right->right->right->rightMax);
				tempNode->right->leftmin = Min(tempNode->right->left->leftmin,tempNode->right->right->leftmin);
				tempNode->right->rightMax = Max(tempNode->right->left->rightMax,tempNode->right->right->rightMax);
				tempNode->right->right->measure = setNodeMeasure(tempNode->right->right);
				tempNode->right->measure = setNodeMeasure(tempNode->right);

				left_rotation(tempNode);
				tempNode->left->rightInterval= tempNode->key;
				tempNode->left->leftInterval= tempNode->leftInterval;
				tempNode->left->leftmin = Min(tempNode->left->left->leftmin,tempNode->left->right->leftmin);
				tempNode->left->rightMax = Max(tempNode->left->left->rightMax,tempNode->left->right->rightMax);
				tempNode->leftmin = Min(tempNode->left->leftmin,tempNode->right->leftmin);
				tempNode->rightMax = Max(tempNode->left->rightMax,tempNode->right->rightMax);
				tempNode->left->measure = setNodeMeasure(tempNode->left);
				tempNode->measure = setNodeMeasure(tempNode);

				tempHeight = tempNode->right->right->height;
				tempNode->left->height = tempHeight + 1;
				tempNode->right->height = tempHeight + 1;
				tempNode->height = tempHeight + 2;
			}
		}
		else
		{
			if (tempNode->left->height > tempNode->right->height)
			{
				tempNode->height = tempNode->left->height + 1;
			}
			else
			{
				tempNode->height = tempNode->right->height + 1;
			}
		}
		if (tempNode->height == old_height)
		{
			finished = 1;
		}
	}
	return (0);
}

m_tree_t * get_node()
{
	m_tree_t *temp;
	nodes_taken += 1;
	if (free_list != NULL )
	{
		temp = free_list;
		free_list = free_list->right;
	}
	else
	{
		if (currentblock == NULL || size_left == 0)
		{
			currentblock = (m_tree_t *) malloc(BLOCKSIZE * sizeof(m_tree_t));
			size_left = BLOCKSIZE;
		}
		temp = currentblock++;
		size_left -= 1;
	}
	return (temp);
}

void return_node(m_tree_t *node)
{
	node->right = free_list;
	free_list = node;
	nodes_returned += 1;
}

//Change
int setNodeMeasure(m_tree_t *tempNode)
{
	int l = tempNode->leftInterval;
	int r = tempNode->rightInterval;

	if ( (tempNode->right->leftmin < l ) && (tempNode->left->rightMax >= r ) )
	{
		tempNode->measure = r - l;
	}
	else if ( (tempNode->right->leftmin < l) && (tempNode->left->rightMax < r) )
	{
		tempNode->measure = tempNode->right->measure + (tempNode->key - l);
	}
	else if ( (tempNode->right->leftmin >= l) && (tempNode->left->rightMax < r) )
	{
		tempNode->measure = tempNode->right->measure + tempNode->left->measure;
	}
	else if ( (tempNode->right->leftmin >= l) && (tempNode->left->rightMax >= r) )
	{
		tempNode->measure = (r - tempNode->key) + tempNode->left->measure;
	}
	else
	{
		printf("None of the 4 cases!\n");
		exit(-1);
	}
	return tempNode->measure;
}

int setLeafMeasure(m_tree_t *tree)
{
	int intervalStart , intervalEnd;
	int l = tree->leftInterval;
	int r = tree->rightInterval;
	intervalStart = intervalEnd = 0;

	if (tree->leftmin < l)
	{
		intervalStart = l;
	}
	if ( (tree->rightMax <= r) && (tree->rightMax >= l) )
	{
		intervalEnd = tree->rightMax;
	}
	if (tree->rightMax > r)
	{
		intervalEnd = r;
	}
	if ( (l <= tree->leftmin) && (tree->leftmin <= r) )
	{
		intervalStart = tree->leftmin;
	}

	tree->measure = intervalEnd - intervalStart;
	return tree->measure;
}

void treeFixer()
{
	int i;
	for (i = numTraversals - 1; i >= 0; i--)
	{
		traversalHistory[i]->leftmin = Min(traversalHistory[i]->left->leftmin,traversalHistory[i]->right->leftmin);
		traversalHistory[i]->rightMax = Max(traversalHistory[i]->left->rightMax,traversalHistory[i]->right->rightMax);
		traversalHistory[i]->measure = setNodeMeasure(traversalHistory[i]);
	}
}

int insert_balanced(m_tree_t *tree, key_t new_key,object_t *new_object, int other_key, int direction)
{
	m_tree_t *tmp_node;
	if (tree->left == NULL )
	{
		tree->left = (m_tree_t *) new_object;
		tree->key = new_key;
		tree->right = NULL;
		tree->leftInterval = NEGATIVE_INFINITY;
		tree->rightInterval = INFINITY;

		tree->list = (interval_node *) malloc(sizeof(interval_node));
		tree->list->rightInterval = other_key;
		tree->list->leftInterval = new_key;


		tree->leftmin = new_key;
		tree->rightMax = other_key;
		tree->measure = setLeafMeasure(tree);
		tree->list->next = NULL;
	}
	else
	{
		numTraversals = 0;
		tmp_node = tree;
		while (tmp_node->right != NULL )
		{
			traversalHistory[numTraversals] = tmp_node;
			numTraversals++;
			if (new_key < tmp_node->key)
			{
				tmp_node = tmp_node->left;
			}
			else
			{
				tmp_node = tmp_node->right;
			}
		}
		if (tmp_node->key == new_key)
		{
			interval_node *temp, *newnode;
			temp = tmp_node->list;
			newnode = (interval_node*) malloc(sizeof(interval_node));
			if (direction == 1)
			{
				newnode->rightInterval = other_key;
				newnode->leftInterval = new_key;
			}
			else
			{
				newnode->rightInterval = new_key;
				newnode->leftInterval = other_key;
			}

			tmp_node->list = newnode;
			newnode->next = temp;

			if (newnode->leftInterval < tmp_node->leftmin)
			{
				tmp_node->leftmin = newnode->leftInterval;
			}
			if (newnode->rightInterval > tmp_node->rightMax)
			{
				tmp_node->rightMax = newnode->rightInterval;
			}
			tmp_node->measure = setLeafMeasure(tmp_node);
			treeFixer();
		}
		else
		{
			m_tree_t *old_leaf, *new_leaf;
			old_leaf = get_node();
			old_leaf->left = tmp_node->left;
			old_leaf->key = tmp_node->key;
			old_leaf->right = NULL;
			old_leaf->height = 0;
			old_leaf->list = tmp_node->list;
			old_leaf->leftmin = tmp_node->leftmin;
			old_leaf->rightMax = tmp_node->rightMax;

			old_leaf->leftInterval = tmp_node->leftInterval;
			old_leaf->rightInterval = tmp_node->rightInterval;

			old_leaf->measure = tmp_node->measure;

			new_leaf = get_node();
			new_leaf->left = (m_tree_t *) new_object;
			new_leaf->key = new_key;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			new_leaf->list = (interval_node*) malloc(sizeof(interval_node));
			new_leaf->list->next = NULL;

			if (direction == 1)
			{
				new_leaf->list->leftInterval = new_key;
				new_leaf->list->rightInterval = other_key;
				new_leaf->leftmin = new_key;
				new_leaf->rightMax = other_key;
			}
			else
			{
				new_leaf->list->leftInterval = other_key;
				new_leaf->list->rightInterval = new_key;
				new_leaf->leftmin = other_key;
				new_leaf->rightMax = new_key;
			}

			if (tmp_node->key < new_key)
			{
				tmp_node->left = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = new_key;
				old_leaf->rightInterval = tmp_node->key;
				new_leaf->leftInterval = tmp_node->key;
				new_leaf->rightInterval = tmp_node->rightInterval;
			}
			else
			{
				tmp_node->left = new_leaf;
				tmp_node->right = old_leaf;
				old_leaf->leftInterval = tmp_node->key;
				new_leaf->leftInterval = tmp_node->leftInterval;
				new_leaf->rightInterval = tmp_node->rightInterval;
			}
			new_leaf->measure = setLeafMeasure(new_leaf);
			old_leaf->measure = setLeafMeasure(old_leaf);

			tmp_node->leftmin = Min(tmp_node->left->leftmin,tmp_node->right->leftmin);
			tmp_node->rightMax = Max(tmp_node->right->rightMax,tmp_node->left->rightMax);

			tmp_node->measure = setNodeMeasure(tmp_node);
			tmp_node->height = 1;
			treeFixer();
			balanceTree();
		}
	}
	return 0;
}

object_t *delete_balanced(m_tree_t *tree, key_t delete_key, int other_key, int direction)
{
	m_tree_t *tmp_node, *upper_node, *other_node;
	interval_node *temp,*prev, *del, *prevdel;
	object_t *deleted_object;
	int minNode, maxNode;
	if (tree->left == NULL )
	{
		return (NULL );
	}
	else if (tree->right == NULL )
	{
		if (tree->key == delete_key)
		{
			deleted_object = (object_t *) tree->left;
			tree->left = NULL;
			tree->measure = 0;
			tree->list = NULL;
			return (deleted_object);
		}
		else
		return (NULL );
	}
	else
	{
		tmp_node = tree;
		numTraversals = 0;
		while (tmp_node->right != NULL )
		{
			traversalHistory[numTraversals] = tmp_node;
			//printf("\nState of lists at this stage - ");
			//printTreeList(tmp_node);
			numTraversals++;
			upper_node = tmp_node;
			if (delete_key < tmp_node->key)
			{
				tmp_node = upper_node->left;
				other_node = upper_node->right;
			}
			else
			{
				tmp_node = upper_node->right;
				other_node = upper_node->left;
			}
			//printf("\nState of lists at this stage - ");
			//printTreeList(tmp_node);
		}
		if (tmp_node->key != delete_key)
		{
			return (NULL );
		}
		else
		{
			int found,count;
			found = count = 0;
			temp = tmp_node->list;

			prev = del = prevdel = NULL;

			while (temp != NULL )
			{
				count++;

				if (direction == 1)
				{
					if ((temp->leftInterval == delete_key) && (temp->rightInterval == other_key))
					{
						prevdel = prev;
						del = temp;

						found = 1;
					}
				}
				else
				{
					if ((temp->leftInterval == other_key) && (temp->rightInterval == delete_key))
					{
						prevdel = prev;
						del = temp;
						found = 1;
					}
				}

				prev = temp;
				temp = temp->next;
			}
			if (found == 0)
			{
				return 0;
			}
			if (count == 1)
			{
				upper_node->key = other_node->key;
				upper_node->left = other_node->left;
				upper_node->right = other_node->right;

				upper_node->leftmin = other_node->leftmin;
				upper_node->rightMax = other_node->rightMax;

				upper_node->list = other_node->list;
				upper_node->height = other_node->height;

				if(other_node->right != NULL)
				{
					upper_node->left->leftInterval = upper_node->leftInterval;
					upper_node->left->measure = setLeafMeasure(upper_node->left);
					upper_node->right->rightInterval = upper_node->rightInterval;
					upper_node->right->measure = setLeafMeasure(upper_node->right);
					upper_node->measure = setNodeMeasure(upper_node);
				}
				else
				{
					upper_node->measure = setLeafMeasure(upper_node);
				}
				numTraversals--;
				treeFixer();
				balanceTree();

				deleted_object = (object_t *) tmp_node->left;
				return_node(tmp_node);
				return_node(other_node);
				return (deleted_object);
			}
			if (prevdel != NULL )
			{
				prevdel->next = del->next;
			}
			else
			{
				tmp_node->list = del->next;
			}
			temp = tmp_node->list;
			minNode = tmp_node->list->leftInterval;
			maxNode = tmp_node->list->rightInterval;
			while(temp != NULL)
			{
				minNode = Min(minNode,temp->leftInterval);
				maxNode = Max(maxNode,temp->rightInterval);
				temp = temp->next;
			}
			tmp_node->leftmin = minNode;
			tmp_node->rightMax = maxNode;
			tmp_node->measure = setLeafMeasure(tmp_node);
			treeFixer();

			/*
				tmp_node->rightmax = setRightMax(tmp_node);
				tmp_node->leftmin = setLeftMin(tmp_node);
				tmp_node->measure = setMeasure(tmp_node);
			}*/
		}
	}
	return (object_t *) 1;
}

void remove_tree(m_tree_t *tree)
{
	m_tree_t *current_node, *tmp;
	if (tree->left == NULL )
	return_node(tree);
	else
	{
		current_node = tree;
		while (current_node->right != NULL )
		{
			if (current_node->left->right == NULL )
			{
				return_node(current_node->left);
				tmp = current_node->right;
				return_node(current_node);
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
		return_node(current_node);
	}
}

int Max(int a, int b)
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

int Min(int a, int b)
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

int setRightMax(m_tree_t *node)
{
	return Max(node->left->rightMax, node->right->rightMax);
}

int setLeftMin(m_tree_t *node)
{
	return Min(node->left->leftmin, node->right->leftmin);
}

#ifndef STEP
int main()
{
	int i;
	m_tree_t *t;
	setbuf(stdout,0);
	printf("starting \n");
	t = create_m_tree();
	for(i=0; i< 50; i++ )
	{
		insert_interval( t, 2*i, 2*i +1 );
	}
	printf("inserted first 50 intervals, total length is %d, should be 50.\n", query_length(t));
	insert_interval( t, 0, 100 );
	printf("inserted another interval, total length is %d, should be 100.\n", query_length(t));
	for(i=1; i< 50; i++ )
	{
		insert_interval( t, 199 - (3*i), 200 ); /*[52,200] is longest*/
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
	insert_interval( t, 2000+i, 3000+i );
	printf("inserted 3000 intervals, total length is %d, should be 4100.\n", query_length(t));
	for(i=0; i<=3000; i++ )
	delete_interval( t, 2000+i, 3000+i );
	printf("deleted 3000 intervals, total length is %d, should be 100.\n", query_length(t));
	for(i=0; i<=100; i++ )
	insert_interval( t, 10*i, 10*i+100 );
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

