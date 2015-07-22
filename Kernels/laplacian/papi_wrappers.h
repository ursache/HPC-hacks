// Author: gilles.fourestey@cscs.ch
//
#ifdef __cplusplus
extern "C"
{
#endif

#include <papi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


typedef char* string;

int EventSet = PAPI_NULL;
static int Events[128];
static string sEvents[128];

static int NUMEVENTS = 0;
static int NOEVENT   = 0;
static long_long values[128];

static int setup = 0;
static int papi_error;


#define PAPI_START papi_start_counters();
#define PAPI_STOP  papi_stop_counters();
#define PAPI_PRINT papi_print_results();


static inline unsigned long long rdtscp() {
  unsigned long long u;
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "%rcx");
  return u;
}


void papi_print_error(int ierr)
{
        char errstring[PAPI_MAX_STR_LEN];
        //PAPI_perror(ierr, errstring, PAPI_MAX_STR_LEN );
        PAPI_perror(errstring);

        printf("PAPI error %s\n", errstring );

}

void papi_setup()
{


        /* Initialize the library */
        papi_error = PAPI_library_init(PAPI_VER_CURRENT);
        if (papi_error != PAPI_VER_CURRENT) 
        {
                printf("PAPI library init error!\n");
                exit(1);
        }

#ifdef _OPENMP
        papi_error =  PAPI_thread_init((long unsigned int (*)()) omp_get_thread_num);
        if ( papi_error != PAPI_OK )
    {
                printf("Could not initialize the library with openmp.\n");
                exit(1);
        }
#endif

        int num_hwcntrs, i;

        papi_error = num_hwcntrs = PAPI_num_counters();
        if (papi_error <= PAPI_OK)
                papi_print_error(papi_error);
        //
        string papi_debug = getenv("PAPI_DEBUG");
        //
        int debug = papi_debug == NULL;
        //
        if (debug)
        {
        printf("This system has %d available counters.\n", num_hwcntrs);
        printf("We will count %d events.\n", NUMEVENTS);
        }

    string papi_counters = getenv("PAPI_EVENTS");
        if (debug)
        printf("PAPI_EVENTS = %s\n", papi_counters);

    string result = NULL;
    char  delim[] = "|";

    result = strtok( papi_counters, delim );

    while( result != NULL )
    {
        //printf( "result is \"%s\"\n", result );
        //strcpy(sEvents[count], result);   
        sEvents[NUMEVENTS] = result;
        NUMEVENTS++;
        result = strtok( NULL, delim );
    }

    if (NUMEVENTS == 0)
    {
                if (debug)
                printf("No event selected\n");

                setup = 1;
        return;
    }


    if (NUMEVENTS > 127)
    {
        printf("Too many events selected\n");
        exit(-1);
    }

        papi_error = PAPI_create_eventset(&EventSet);

        if (papi_error != PAPI_OK)
                printf("Could not create the EventSet: %d\n", papi_error);


    for (i = 0; i < NUMEVENTS; i++)
        printf("Event %d out of %d = %s\n", i, NUMEVENTS, sEvents[i]);

        for (i = 0; i < NUMEVENTS; i++)
        {
        if (PAPI_event_name_to_code(sEvents[i], &Events[i]) != PAPI_OK)
        {
            printf("Event %s not recognised\n", sEvents[i]);
        }

                papi_error = PAPI_add_event(EventSet, Events[i]);
                if (papi_error != PAPI_OK)
                {
                        printf("Could not add event %s to the event set: %d, %d\n", sEvents[i], papi_error, PAPI_OK);
                }
  }
        setup = 1;
}


void papi_start_counters()
{
        if (!setup) papi_setup();
        if (NUMEVENTS == 0) return;

        papi_error = PAPI_start(EventSet);
        if (papi_error != PAPI_OK) papi_print_error(papi_error);
}


void papi_stop_counters()
{
        if (NUMEVENTS == 0) return;

        papi_error = PAPI_stop(EventSet, values);
        if (papi_error != PAPI_OK) papi_print_error(papi_error);
}

void papi_print_results()
{
        if (NUMEVENTS == 0) return;

        char eventName[PAPI_MAX_STR_LEN];
        int i;

        for(i = 0; i < NUMEVENTS; i++)
        {
                papi_error = PAPI_event_code_to_name(Events[i], eventName);
                if (papi_error != PAPI_OK) papi_print_error(papi_error);
                printf("Event %s: %lu\n", eventName, values[i]);
        }
}

#ifdef __cplusplus
}
#endif

