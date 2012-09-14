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
	for( i = 2; i <95; i++ )
		insert_line(txt1, 2, "some filler line between 1 and 95" );
	if( (tmp = length_text(txt1)) != 100)
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
