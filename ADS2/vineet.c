#include <stdio.h>
#include <stdlib.h>

#define NEGATIVE_INFINITY -102400
#define INFINITY 10240000

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
	struct tr_n_t *left, *right;
	int height;
	int leftmin, rightMax;
	int measure;
	int leftInterval, rightInterval;
	interval_node *list;
} m_tree_t;

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
int setMeasure(m_tree_t *node);
m_tree_t * get_node();

int insert_balanced(m_tree_t *tree, key_t new_key,object_t *new_object, int other_key, int direction);
int delete_balanced(m_tree_t *tree, key_t currEndpoint, int otherEndpoint, int direction);
void printTreeList(m_tree_t *node);
void printList(interval_node *node);
void insert_interval(m_tree_t *tree, int leftKey, int rightKey);
int getLength(interval_node *list);
void fixList(unsigned int key,interval_node *this_list,interval_node *other_list);
int intervalInList(interval_node *list,int leftInterval,int rightInterval);
interval_node  *deleteFromList(interval_node *list, int leftInterval, int rightInterval);
void recursiveFixxer(m_tree_t *root);
interval_node *addToList(interval_node *list, unsigned int leftInterval,unsigned int rightInterval);


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

	n->left->rightInterval= n->key;
	n->left->leftInterval= n->leftInterval;
	n->left->leftmin = Min(n->left->left->leftmin,n->left->right->leftmin);
	n->left->rightMax = Max(n->left->left->rightMax,n->left->right->rightMax);
	n->leftmin = Min(n->left->leftmin,n->right->leftmin);
	n->rightMax = Max(n->left->rightMax,n->right->rightMax);
	n->left->measure = setNodeMeasure(n->left);
	n->measure = setNodeMeasure(n);

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
	n->right->leftInterval= n->key;
	n->right->rightInterval= n->rightInterval;

	n->right->leftmin = Min(n->right->left->leftmin,n->right->right->leftmin);
	n->right->rightMax = Max(n->right->left->rightMax,n->right->right->rightMax);

	n->leftmin = Min(n->left->leftmin,n->right->leftmin);
	n->rightMax = Max(n->left->rightMax,n->right->rightMax);

	n->right->height = n->right->left->height + 1;
	n->height = n->right->height + 1;
	n->right->measure = setNodeMeasure(n->right);
	n->measure = setNodeMeasure(n);
}

//Integrate
void balanceTree()
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
			}
			else
			{
				left_rotation(tempNode->left);

				right_rotation(tempNode);

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
				tempNode->left->height = tempNode->left->right->height + 1;
				tempNode->height = tempNode->left->height + 1;
			}
			else
			{
				right_rotation(tempNode->right);

				left_rotation(tempNode);

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
	return;
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

int insert_balanced(m_tree_t *tree, key_t currIntervalEndpoint,object_t *new_object, int otherIntervalEndpoint, int direction)
{
	m_tree_t *tmp_node;
	if (tree->left == NULL )
	{
		tree->left = (m_tree_t *) new_object;
		tree->key = currIntervalEndpoint;
		tree->right = NULL;
		tree->leftInterval = NEGATIVE_INFINITY;
		tree->rightInterval = INFINITY;

		tree->list = (interval_node *) malloc(sizeof(interval_node));
		tree->list->rightInterval = otherIntervalEndpoint;
		tree->list->leftInterval = currIntervalEndpoint;


		tree->leftmin = currIntervalEndpoint;
		tree->rightMax = otherIntervalEndpoint;
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
			if (currIntervalEndpoint < tmp_node->key)
			{
				tmp_node = tmp_node->left;
			}
			else
			{
				tmp_node = tmp_node->right;
			}
		}
		if (tmp_node->key == currIntervalEndpoint)
		{
			interval_node *temp, *newnode;
			temp = tmp_node->list;
			newnode = (interval_node*) malloc(sizeof(interval_node));
			if (direction == 1)
			{
				newnode->rightInterval = otherIntervalEndpoint;
				newnode->leftInterval = currIntervalEndpoint;
			}
			else
			{
				newnode->rightInterval = currIntervalEndpoint;
				newnode->leftInterval = otherIntervalEndpoint;
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
			new_leaf->key = currIntervalEndpoint;
			new_leaf->right = NULL;
			new_leaf->height = 0;
			new_leaf->list = (interval_node*) malloc(sizeof(interval_node));
			new_leaf->list->next = NULL;

			if (direction == 1)
			{
				new_leaf->list->leftInterval = currIntervalEndpoint;
				new_leaf->list->rightInterval = otherIntervalEndpoint;

				new_leaf->leftmin = currIntervalEndpoint;
				new_leaf->rightMax = otherIntervalEndpoint;
			}
			else
			{
				new_leaf->list->leftInterval = otherIntervalEndpoint;
				new_leaf->list->rightInterval = currIntervalEndpoint;

				new_leaf->leftmin = otherIntervalEndpoint;
				new_leaf->rightMax = currIntervalEndpoint;
			}

			if (tmp_node->key < currIntervalEndpoint)
			{
				tmp_node->left = old_leaf;
				tmp_node->right = new_leaf;
				tmp_node->key = currIntervalEndpoint;

				new_leaf->leftInterval = tmp_node->key;
				new_leaf->rightInterval = tmp_node->rightInterval;

				old_leaf->rightInterval = tmp_node->key;

			}
			else
			{
				tmp_node->left = new_leaf;
				tmp_node->right = old_leaf;

				new_leaf->leftInterval = tmp_node->leftInterval;
				new_leaf->rightInterval = tmp_node->rightInterval;

				old_leaf->leftInterval = tmp_node->key;
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

int delete_balanced(m_tree_t *tree, key_t currEndpoint, int otherEndpoint, int direction)
{
	m_tree_t *tmp_node, *upper_node, *other_node;
	interval_node *temp,*prev, *del, *prevdel;
	int minNode, maxNode;

	if (tree->left == NULL )
	{
		return (NULL );
	}
	else if (tree->right == NULL )
	{
		if (tree->key == currEndpoint)
		{
			tree->left = NULL;
			tree->measure = 0;
			tree->list = NULL;
			return 1;
		}
		else
		{
			return 0;
		}
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
			if (currEndpoint < tmp_node->key)
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
		if (tmp_node->key != currEndpoint)
		{
			return (NULL );
		}
		else
		{
			int length, foundInterval;
			foundInterval = length = 0;
			temp = tmp_node->list;

			prev = del = prevdel = NULL;

			while (temp != NULL )
			{
				length++;

				if (direction == 2)
				{
					if ((temp->leftInterval == otherEndpoint) && (temp->rightInterval == currEndpoint))
					{
						prevdel = prev;
						del = temp;

						foundInterval = 1;
					}
				}
				else if (direction == 1)
				{
					if ((temp->leftInterval == currEndpoint) && (temp->rightInterval == otherEndpoint))
					{
						prevdel = prev;
						del = temp;

						foundInterval = 1;
					}
				}

				prev = temp;
				temp = temp->next;
			}
			if (foundInterval == 0)
			{
				return 0;
			}

			/*if (direction == 2)
			{
				if (!intervalInList(tmp_node->list,otherEndpoint,currEndpoint))
				{
					return 0;
				}
			}
			else if (direction == 1)
			{
				if (!intervalInList(tmp_node->list,currEndpoint,otherEndpoint))
				{
					return 0;
				}
			}*/
			//length = getLength(tmp_node->list);
			if (length == 1)
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
					upper_node->right->rightInterval = upper_node->rightInterval;
					upper_node->right->measure = setLeafMeasure(upper_node->right);

					upper_node->left->leftInterval = upper_node->leftInterval;
					upper_node->left->measure = setLeafMeasure(upper_node->left);
					upper_node->measure = setNodeMeasure(upper_node);
				}
				else
				{
					upper_node->measure = setLeafMeasure(upper_node);
				}
				numTraversals--;

				treeFixer();
				balanceTree();

				return_node(tmp_node);
				return_node(other_node);
				return 1;
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
			/*if (direction == 2)
			{
				temp = deleteFromList(tmp_node->list,otherEndpoint,currEndpoint);
			}
			else if (direction == 1)
			{
				temp = deleteFromList(tmp_node->list,currEndpoint,otherEndpoint);
			}*/

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
	return 1;
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
	return 0;
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

int getLength(interval_node *list)
{
	interval_node *temp;
	int length = 0;
	temp = list;
	while (temp != NULL)
	{
		temp = temp->next;
		length++;
	}
	return length;
}

int intervalInList(interval_node *list,int leftInterval,int rightInterval)
{
	interval_node *temp;
	temp = list;
	while (temp != NULL)
	{
		if ( (temp->leftInterval == leftInterval) && (temp->rightInterval == rightInterval))
		{
			return 1;
		}
		temp = temp->next;
	}
	return 0;
}

interval_node *deleteFromList(interval_node *list, int leftInterval,int rightInterval)
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
				//free(temp);
				//printf("\nList after deleting (head case) - \n");
				//printList(list);
				return list;
			}
			else
			{
				prev->next = temp->next;
				//free(temp);
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
		root->rightMax = setRightMax(root);
		root->leftmin =  setLeftMin(root);
		root->measure =  setMeasure(root);

		fixList(root->key,root->list,root->right->list);
		fixList(root->key,root->list,root->left->list);
	}
	else if (root->height == 1)
	{
		root->rightMax = setRightMax(root);
		root->leftmin =  setLeftMin(root);

		//Who will fix the fixxer?
		if (root->leftInterval < root->leftmin)
			root->leftInterval = root->leftmin;

		if (root->rightInterval > root->rightMax)
			root->leftInterval = root->rightMax;

		root->measure =  setMeasure(root);

		root->left->leftInterval = root->leftInterval;
		root->left->rightInterval = root->key;

		root->right->rightInterval = root->rightInterval;
		root->right->leftInterval = root->key;

	}
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

 int setMeasure(m_tree_t *node)
{
	int l = node->leftInterval;
	int r = node->rightInterval;
	int measure = 1;
	//if (node->height > 1 )
	//{
		if (node->right->leftmin < l && node->left->rightMax >= r)
		{
			measure = r - l;
		}
		else if (node->right->leftmin >= l && node->left->rightMax >= r)
		{
			measure = (r - node->key) + node->left->measure;
		}
		else if (node->right->leftmin < l && node->left->rightMax < r)
		{
			measure = node->right->measure + (node->key - l);
		}
		else if (node->right->leftmin >= l && node->left->rightMax < r)
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


