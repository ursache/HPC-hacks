// 
// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#ifdef MPI
#include <mpi.h>
#endif

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


#define DIM_N 1000
#define DIM_M 1000
#define DIM_K 1000

#define NREP  1
//
void laplacian(double*, double*, int, int, int);
//
static double EPSI=1.0e-2;
//
int main(int argc, char** argv) 
{
	int dim_n = DIM_N;
	int dim_m = DIM_M;
	int dim_k = DIM_K;
	//      
	if (argc == 4)
	{
		dim_n = atoi(argv[1]);
		dim_m = atoi(argv[2]);
		dim_k = atoi(argv[3]);
	}
	int local_k;
	//
#ifdef MPI
	MPI_Init(&argc, &argv);
	int num_tasks;
	int my_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	local_k = dim_k/num_tasks; 
#endif
	//
	double* restrict storage1 = (double*)malloc(dim_n*dim_m*dim_k*sizeof(double));
	double* restrict storage2 = (double*)malloc(dim_n*dim_m*dim_k*sizeof(double));

	printf("3x3 stencil: M=%d N=%d K=%d global.\n", dim_n, dim_m, dim_k);

	printf("array size = %f MB\n", (double) dim_n*dim_m*dim_k*sizeof(double)/1024/1024);

	double alloctime = -myseconds();
	init (storage1, dim_n, dim_m, dim_k);
	init (storage2, dim_n, dim_m, dim_k);
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
		double ftime = -myseconds();
		//PAPI_START;
		unsigned long long c0 = cycles();
		//
		for (ii = 0; ii < nrep; ++ii)
		{
			laplacian(st_read, st_write, dim_m, dim_n, dim_k);
		}
		//
		unsigned long long c1 = cycles();
		//unsigned long long cycles = (c1 - c0)/((unsigned long long) (dim_m - 1)*(dim_n - 1));
		unsigned long long cycles = (c1 - c0)/nrep;
		//
		//PAPI_STOP;
		ftime += myseconds();
		ftime /= nrep;
		//print(st_write + dim_n*dim_m, dim_m, dim_n);
		//PAPI_PRINT;
		//
		norminf = maxNorm(st_write, st_read, dim_n*dim_m*dim_k);
		norml2   = l2Norm(st_write, st_read, dim_n*dim_m*dim_k);
		double flops = (dim_m - 2)*(dim_n - 2)*(dim_k - 2)*6.;
		//
		printf("iter = %d, %f s. %ld c., linf = %g l2 = %g, %d flops, %ld c, %f F/c, %f GF/s\n", n_iters, ftime, cycles, norminf, norml2, (long long int) flops, flops/cycles, (double) flops/ftime/1e9); 
	} while (norminf > EPSI);
	time += myseconds();
	//
	printf("\n# iter= %d time= %g residual= %g\n", n_iters, time, norml2); 
	//
	free(storage1);
	free(storage2);
	//
#ifdef MPI
	MPI_Finalize();
#endif
	return 0;
}
