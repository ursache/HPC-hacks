#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>

#include "utils.h"
//#include "cscs_papi.h"
// IACA stuff
#ifdef IACA
#include "iacaMarks.h"
#else
#define IACA_START 
#define IACA_END 
#endif

#ifdef PAPI
#include "papi_wrappers.h"
#endif


#define MIN(a,b)  (((a<b)?a:b))

#define DIM_N 1000
#define DIM_M 1000

#define NREP  1
//
void laplacian(double*, double*, int, int);
//
void init(double*  p, int n, int m)
{
#pragma omp parallel for
	for (int j = 0;  j < n; ++j) 
	{
		for (int i = 0; i < n; i++) 
		{

			if ( ((i == 0) || (j == 0)) || (i == n-1) || (j == m-1) )
				p[j*n + i] = 1.;
			else
				p[j*n + i] = 0.;
		}
	}
}


double maxNorm(double* v1, double* v2, int size)
{

	double mymax = 0.;
#pragma omp parallel for reduction(max:mymax) 
	for (int ii = 0; ii < size; ++ii)
	{
		if (fabs(*v1 - *v2) > mymax)
		{
			//printf("i = %d, v1 = %g, v2 = %g, max = %g\n", ii, *v1, *v2, fabs(*v1 - *v2));
			mymax = fabs(*v1 - *v2);
		}
		++v1; ++v2;
	}
	return mymax;
}


double l2Norm(double* v1, double* v2, int size)
{

        double myl2 = 0.;
#pragma omp parallel for reduction(+:myl2) 
        for (int ii = 0; ii < size; ++ii)
        {
		myl2 += fabs(v1[ii]-v2[ii])*(v1[ii] - v2[ii]);
        }
        return sqrt(myl2)/size;
}




void print(double* p, int m, int n)
{
	for (int i=0; i < MIN(n, 15); ++i) 
	{
		for (int j=MIN(m, 15); j > 0; --j) 
		{
			printf("%e ", *p);
			++p;
		}
		p += m - MIN(m, 15);
		printf("\n");
	}
}

static double EPSI=1.0e-1;

int main(int argc, char** argv) 
{

	int dim_n = DIM_N;
	int dim_m = DIM_M;
	//      
	if (argc == 3)
	{
		dim_n = atoi(argv[1]);
		dim_m = atoi(argv[2]);
	}

	double* restrict storage1 = (double*)malloc(dim_n*dim_m*sizeof(double));
	double* restrict storage2 = (double*)malloc(dim_n*dim_m*sizeof(double));

	printf("3x3 stencil...%dx%d\n\n", dim_m, dim_n);
	printf("array sizes = %f MB\n", (double) dim_n*dim_m*sizeof(double)/1024/1024);

	double alloctime = -myseconds();
	init (storage1, dim_n, dim_m);
	init (storage2, dim_n, dim_m);
	alloctime += myseconds();
	printf("Allocation time = %f s.\n\n", alloctime);

	int n_iters = 0;
	int nrep    = NREP;
	int ii;

	double * st_read  = storage2;
	double * st_write = storage1;

	double phi;
	double norminf, norml2;
	double time = - myseconds();
	int count = 0;	
	//
	do 
	{
		count++;
		// swaping the solutions
		double *tmp = st_read;
		st_read     = st_write;
		st_write    = tmp;
		//
		++n_iters;
		//
		//#pragma omp parallel for 
		//#pragma acc parallel loop
		double ftime = -myseconds();
		//PAPI_START;
		unsigned long long c0 = cycles();
		//
		for (ii = 0; ii < nrep; ++ii)
		{
			laplacian(st_read, st_write, dim_m, dim_n);
		}
		//
		unsigned long long c1 = cycles();
		//unsigned long long cycles = (c1 - c0)/((unsigned long long) (dim_m - 1)*(dim_n - 1));
		unsigned long long cycles = (c1 - c0)/nrep;
		//
		//PAPI_STOP;
		ftime += myseconds();
		ftime /= nrep;
		//print(st_write, dim_m, dim_n);
		//PAPI_PRINT;
		//
		norminf = maxNorm(st_write, st_read, dim_n*dim_m);
		norml2   = l2Norm(st_write, st_read, dim_n*dim_m);
		double flops = (dim_m - 2)*(dim_n - 2)*4.;
		//
		printf("iter = %d, linf norm = %g l2 norm = %g, %d flops, %ld cycles, %f flops/cycle, %f s., %f Gflops/s\n", n_iters, norminf, norml2, (int) flops, cycles, flops/cycles, ftime, (double) flops/ftime/1e9); 
	// while (norml2 > EPSI);
	} while (count < 20);

	time += myseconds();

	printf("\n# iter= %d time= %g residual= %g\n", n_iters, time, norml2); 

	free(storage1);
	free(storage2);

	return 0;
}
