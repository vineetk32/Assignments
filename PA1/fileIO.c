#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc,char **argv)
{
	int rank, retCode = 0;
	double *buffer;
	MPI_File handle;
	MPI_Status status;
	int i = 0,j= 0,k = 0;
	FILE *fout;

	if (argc != 3)
	{
		printf("\nInvalid no. of arguments.\n\nUsage - fileConverter.x <gpfs_file> <outfilename>\n\n");
		return 0;
	}
	retCode = MPI_Init(&argc,&argv);
	if (retCode != MPI_SUCCESS)
	{
		printf("\nError during init! RetCode - %d\n",retCode);
		MPI_Abort(MPI_COMM_SELF,retCode);
	}
	MPI_Comm_rank(MPI_COMM_SELF,&rank);

	buffer = (double *) malloc(sizeof(double) * 4096 * 4096);


	MPI_File_open(MPI_COMM_SELF,argv[1],MPI_MODE_RDONLY,MPI_INFO_NULL,&handle);

	if (handle != NULL)
	{
		MPI_File_read(handle,buffer,4096 * 4096,MPI_DOUBLE,&status);
		//printf("\nFile read status is - %d",status);
		MPI_File_close(&handle);
		MPI_Finalize();

		fout = fopen(argv[2],"w");
		if (fout != NULL)
		{
			for (j = 0; j < 4096; j++)
			{
				fprintf(fout,"\n");
				for (i = 0; i< 4096; i++)
				{
					fprintf(fout,"%lf ",buffer[j * k + i]);

				}
				k++;
			}
			fclose(fout);
		}
		else
		{
			printf("\n Cant open %s for writing.\n",argv[2]);
		}
	}
	else
	{
		printf("\n Cant open %s for writing.\n",argv[1]);
	}

	return 0;
}
