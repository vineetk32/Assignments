#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {

	int d, n;
	float rbee, rflow;
	FILE *fp;
	int i, j, l;
	float h;

	if( argc != 6 ) {
		printf("generateinput <D> <N> <Rbee> <Rflow> <filename>\n");
		return 1;
	}

	d = atoi(argv[1]);
	n = atoi(argv[2]);
	rbee = atof(argv[3]);
	rflow = atof(argv[4]);

	fp = fopen(argv[5], "w");

	fprintf(fp, "%d %d\n", d, n);
	fprintf(fp, "%f %f\n", rbee, rflow);

	srand( time(NULL) );
	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			l = rand() % 2;
			fprintf(fp, "%d ", l);
		}
		fprintf(fp, "\n");
	}

	for( i = 0; i < d; i++ ) {
		for( j = 0; j < d; j++ ) {
			h = ((float) rand()) / RAND_MAX;
			fprintf(fp, "%f ", h);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);

	return 0;
}
