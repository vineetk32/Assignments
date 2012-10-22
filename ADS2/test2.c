
int main()
{  
	int i; m_tree_t *t; ;
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