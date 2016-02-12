#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef __GNUC__
#include <immintrin.h>
#endif

#include "common.h"
#ifdef PAPI
#include "papi_wrappers.h"
#endif

int main(int argc, char** argv) 
{
	printf("Doing a 3x3 stencil 'by hand'\n\n");
	//
	int dim_n = DIM_N;
	int dim_m = DIM_M;
	//
	if (argc == 2)
	{
		dim_n = dim_m = atoi(argv[1]);
	}
	if (argc == 3)
	{
		dim_n = atoi(argv[1]);
		dim_m = atoi(argv[2]);
	}
	long int size = dim_n*dim_m*sizeof(double);
	//
	printf("Total array allocation: %g MB\n", 2*size/1024/1024.);
	printf("x-direction size      : %f MB\n", dim_m*sizeof(double)/1024/1024.);
	//
	double* storage1 = (double*) _mm_malloc(dim_n*dim_m*sizeof(double), 32);
	if (!storage1)
		printf("Failed allocating storage1\n");
	double* storage2 = (double*) _mm_malloc(dim_n*dim_m*sizeof(double), 32);
	if (!storage2)
		printf("Failed allocating storage2\n");
	//
	printf("Initializing storage1...\n");
	init (storage1, dim_n, dim_m);
	printf("Initializing storage2...\n");
	init (storage2, dim_n, dim_m);
	printf("Init done.\n"); fflush(stdout);

	int n_iters = 0;

	double * st_read  = storage2;
	double * st_write = storage1;

	double norm;
	//
	do 
	{
		// swaping the solutions
		double *tmp = st_read;
		st_read     = st_write;
		st_write    = tmp;
		//   
		++n_iters;
		//
		int i;
		int j;

#ifdef PAPI
		PAPI_START;
#endif
		long int count = 0;
		double time    = 0;
		time = -myseconds();
		for (j = 1; j < dim_m - 1; ++j)
		{
			//
			int i  = 1;
			int is = 1;
			while ((long) (st_write + j*dim_n + is) & (long) 0x1F) is++;
#if 1
			__asm__ volatile ("#first loop");
			for (i = 1; i < is; ++i, ++count)
			{
                               scheme(st_read + j*dim_n + i, st_write + j*dim_n + i, dim_n);
			}
#endif
			__asm__ volatile ("#main loop");
			#pragma ivdep
			#pragma vector aligned
			for (; i + 4 < dim_n - 1; i = i + 4, count = count + 4)
			{
				avx_scheme(st_read + j*dim_n + i, st_write + j*dim_n + i, dim_n); 
			}
			__asm__ volatile ("#last loop");
			#pragma ivdep
			for (; i < dim_n - 2; ++i, ++count)
			{
				
				scheme(st_read + j*dim_n + i, st_write + j*dim_n + i, dim_n);
			}
		}
		time += myseconds();
#ifdef PAPI
		PAPI_STOP;
		PAPI_PRINT;
#endif
		norm = maxNorm(st_write, st_read, dim_n*dim_m);
		//
		long int num_loads  = count*4;
		long int num_stores = count;
		long int flops      = count*4;
		//
		printf("iter = %d, norm = %g, GB/s = %g, Gflops/s = %g, count = %d\n", n_iters, norm, (double) (num_loads + num_stores)*8/time/1024/1024/1024., (double) flops/time/1.e9, count); 
		//
		if (norm < EPSI) break;
		//if (n_iters == 2) break;

	} while (1);

exit:
	print(storage2, dim_n, dim_m);

	printf("\n# iter= %d, residual= %g\n", n_iters, norm); 

	_mm_free(storage1);
	_mm_free(storage2);

	return 0;
}
