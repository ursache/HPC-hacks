#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef __GNUC__
#include <immintrin.h>
#endif

#include "common.h"
#include "papi_wrappers.h"



int main(int argc, char** argv) 
{
	printf("Doing a 3x3 stencil 'by hand'\n\n");
	//
	int dim_n = DIM_N;
	int dim_m = DIM_M;
	//
	if (argc == 3)
	{
		dim_n = atoi(argv[1]);
		dim_m = atoi(argv[2]);
	}
	long int size = dim_n*dim_m*sizeof(double);
	//
	printf("Total array allocation: %g GB\n", 2*size/1024/1024/1024.);
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
		//
		__m128d WEST1;
		__m128d NORTH1;
		__m128d SOUTH1;
		__m128d EAST1;
		//
		__m128d WEST2;
		__m128d NORTH2;
		__m128d SOUTH2;
		__m128d EAST2;
		//
		__m128d WEST3;
		__m128d NORTH3;
		__m128d SOUTH3;
		__m128d EAST3;
		//
		__m128d WEST4;
		__m128d NORTH4;
		__m128d SOUTH4;
		__m128d EAST4;
		//
                __m128d alpha = _mm_set1_pd(0.25);
		//
		time = -myseconds();
		for (j = 1; j < dim_m - 1; ++j)
		{
			//
			int i  = 1;
			int is = 1;
			//printf("%d %p %p\n", is, st_write + j*dim_n + is - 1, st_write + j*dim_n + is);
			while ((long) (st_write + j*dim_n + is) & (long) 0x0F) is++;
			//{printf("%d %p\n", is, st_write + j*dim_n + is);is++;};
			//
			//printf("dim_n = %d, rest = %d, is = %d\n", dim_n - 2, (dim_n - 2)%2, is);
#if 1
			//printf("first i = %d, %d\n", i, is);
			__asm__ volatile ("#first loop");
			for (i = 1; i < is; ++i, ++count)
			{
			       //printf("		i = %d, %d\n", i, is);
                               scheme(st_read + j*dim_n + i, st_write + j*dim_n + i, dim_n);
			}
#endif
			//
			//printf("main: i = %d, %d\n", i, dim_n - 1 - 1);
			__asm__ volatile ("#main loop");
			//
                        double* vl = st_read  + j*dim_n + is;
                        double* vs = st_write + j*dim_n + is;
			//WEST1  = _mm_loadu_pd (vl - 1 );
			#pragma ivdep
			#pragma vector aligned
			for (; i + 4 < dim_n - 1; i = i + 4, count = count + 4)
			//for (;: i < dim_n - 1; i = i + 1, count = count + 1)
			{
				EAST1  = _mm_loadu_pd (vl + 1 );
				WEST1  = _mm_loadu_pd (vl - 1 );
				NORTH1 = _mm_loadu_pd (vl + dim_n);
				SOUTH1 = _mm_loadu_pd (vl - dim_n);
				//
				EAST2  = _mm_loadu_pd (vl + 1 + 2);
				WEST2  = _mm_loadu_pd (vl - 1 + 2);
				NORTH2 = _mm_loadu_pd (vl + dim_n + 2);
				SOUTH2 = _mm_loadu_pd (vl - dim_n + 2);
				/*
				EAST3  = _mm_loadu_pd (vl + 1 + 4);
				WEST3  = _mm_loadu_pd (vl - 1 + 4);
				NORTH3 = _mm_loadu_pd (vl + dim_n + 4);
				SOUTH3 = _mm_loadu_pd (vl - dim_n + 4);
				//
				EAST4  = _mm_loadu_pd (vl + 1 + 6);
				WEST4  = _mm_loadu_pd (vl - 1 + 6);
				NORTH4 = _mm_loadu_pd (vl + dim_n + 6);
				SOUTH4 = _mm_loadu_pd (vl - dim_n + 6);
				*/
				SSE_STORE(vs + 0, alpha*(EAST1 + WEST1 + SOUTH1 + NORTH1));
				SSE_STORE(vs + 2, alpha*(EAST2 + WEST2 + SOUTH2 + NORTH2));
				//_mm_stream_pd(vs + 4, alpha*(EAST3 + WEST3 + SOUTH3 + NORTH3));
				//_mm_stream_pd(vs + 6, alpha*(EAST4 + WEST4 + SOUTH4 + NORTH4));
				//
				//WEST1 = EAST2;
				//
				vl += 4;
				vs += 4;
			}
			//IACA_END;
			//
			//printf("last: i = %d, %d\n", i, dim_n - 1);
			//goto exit;
			__asm__ volatile ("#last loop");
			#pragma ivdep
			for (; i < dim_n - 1; ++i, ++count)
			{
				scheme(st_read + j*dim_n + i, st_write + j*dim_n + i, dim_n);
			}
			//
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
