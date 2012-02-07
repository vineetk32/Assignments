#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>

#define MAX_THREAD_NUM 300
#define MAX_TEXTSIZE (1<<20)
#define MAX_LINE_SIZE 32
#define MAX_PROC 10

char* proc_list[MAX_PROC]; //share the names of the process between the threads
int results[MAX_THREAD_NUM][MAX_PROC]; // write the results in this array
struct th_arg_s
{
    char *buf;
    int start;
    int end;
    int thread_id;
    int count;
};
void* string_match(void* arg)
{
    struct th_arg_s *th_arg = (struct th_arg_s*)(arg);
    //find the start of the line from which one has to read
    int start=th_arg->start;
    int end=0;
    int start_ch =th_arg->buf[th_arg->start];
    while (start_ch!='\n' && start_ch != EOF && (start-th_arg->start)<=th_arg->count && start!=0)
        //check if the first character is newline or eof
        //also verrify that the start character is in range of start + count that was supplied by the main program
    {
        //if(th_arg->buf[(th_arg->start)-1]=='\n') //if the block was divided in such a way that block starts from a new line
        //    break;
        start+=1;;
        start_ch =th_arg->buf[start];

    }
    if (start!=th_arg->start)
        start+=1;
    //we have the start position of the new line
    int lo_in=th_arg->start+th_arg->count;
    while (th_arg->buf[lo_in]!='\n')
    {
        lo_in=lo_in+1;
    }
    end=lo_in;

    //now we have the start and end position of the block that we want to analyze
    int line_start=start;
    if (line_start>=end)
    {
        //if it goes beyond its boundary then exit the thread
        pthread_exit((void*) arg);
    }
    while (1)
    {
        int lo_in2;

        for (lo_in2=0;lo_in2<10 && (proc_list[lo_in2]!=0);lo_in2++)
        {
            if (!strncmp(&(th_arg->buf[line_start+16]),proc_list[lo_in2],strlen(proc_list[lo_in2])))
            {
                results[th_arg->thread_id][lo_in2]++;
            }
        }
        while (line_start<end && th_arg->buf[line_start]!='\n' && th_arg->buf[line_start]!=EOF)
        {
            line_start++;
        }
        line_start++;//to adjust the pointer to the next line
        if (line_start>=end)
            break;
    }
    pthread_exit((void*) arg);
}
int main(int argc, char** argv)
{
    pthread_t thread[MAX_THREAD_NUM];
    int num[MAX_THREAD_NUM];
    int final_count[10]={0},proc_lcount=0;
    struct th_arg_s th_arg[MAX_THREAD_NUM];
    pthread_attr_t attr;
    int rc,t,ptr,log_lines,count;
    struct th_arg_s *status;
    char *buf;
    struct timeval p[MAX_THREAD_NUM],q[MAX_THREAD_NUM];
    int textsize,blocksize;

    //check if the proper inputs ahve been given
    if (argc <3)
    {
        printf("There needs to be two arguments \n");
        exit(1);
    }

    //open both the files and see if they were opened successfully
    FILE *fp_log=fopen(argv[1],"r");
    FILE *fp_proc=fopen(argv[2],"r");
    if (fp_log==NULL || fp_proc==NULL)
    {
        printf("error opening the file \n");
        exit(1);
    }

    //copy the data of the log file in a buffer
    if ( (buf = (char*)malloc(MAX_TEXTSIZE)) == NULL )
        exit(-1);

    for ( ptr = 0 ; !feof(fp_log) ; )
        ptr += fread(&(buf[ptr]), 1, 1, fp_log);

    textsize = ptr;
    //printf("the textsize is %d and the text is %s\n",textsize,buf);

    //parse the process names from the file and store them in an array
    int lo=0; //loop variable
    while (!feof(fp_proc) && lo<10)
    {
        proc_list[lo]=malloc(sizeof(char)*20);
        fscanf(fp_proc,"%s\n",proc_list[lo]);
        //printf("the read value was %s at %d \n",*(proc_list+lo),lo);
        lo++;
    }
    //


    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    //count=25;
    for (count=10;count<=MAX_THREAD_NUM;count=count+5)
    {
        gettimeofday(&p[count%10], NULL);

        for (t=0; t<count; t++)
        {
            th_arg[t].end=textsize;
            th_arg[t].start= t>0 ? (th_arg[t-1].start+th_arg[t-1].count):0;
            th_arg[t].buf=buf;
            th_arg[t].count=(textsize/count)+1;
            th_arg[t].thread_id=t;
            for (lo=0;lo<10;lo++)
            {
                results[t][lo]=0;
            }

            rc = pthread_create(&thread[t], &attr, string_match, (void *)&th_arg[t]);
            if (rc!=0)
            {
                perror("thread_create:");
            }
        }
        /* Free attribute and wait for the other threads */
        pthread_attr_destroy(&attr);
        proc_lcount=0;
        for(t=0;t<10;t++)
        {   final_count[t]=0; //reset the final count
            }
        for (t=0; t<count; t++)
        {
            rc = pthread_join(thread[t], (void**)&status);

            int no_comm=0;

            while (1)
            {
                final_count[no_comm]=final_count[no_comm]+results[t][no_comm];
                proc_lcount=proc_lcount+results[t][no_comm];
                no_comm++;
                if (no_comm>10 || proc_list[no_comm]==0)
                    break;
            }
            if (rc!=0)
            {
                perror("thread_join:");
                exit(-1);
            }
        }
        gettimeofday(&q[count%10], NULL);
        log_lines=0;
        printf("%d \t %d \t %d \t %8ld\n",(textsize/count),count,proc_lcount,q[count%10].tv_usec - p[count%10].tv_usec + (q[count%10].tv_sec-p[count%10].tv_sec)*1000000);
    }
    for (t=0;t<10 && proc_list[t]!=0;t++)
        {
            printf("pName: %s count: %d \n",proc_list[t],final_count[t]);
            log_lines+=final_count[t];
        }
    printf("The total log lines are %d \n",log_lines);

}
