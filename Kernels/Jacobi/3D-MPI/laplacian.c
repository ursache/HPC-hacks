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
	else
	if (argc == 2)
	{
		dim_n = dim_m = dim_k = atoi(argv[1]);
	}
	int K_loc = dim_k;
	//
#ifdef MPI
	//
	MPI_Init(&argc, &argv);
	int num_tasks;
	int my_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0) printf("Hello friend... %d friends\n\n", num_tasks);
	MPI_Barrier(MPI_COMM_WORLD);
	//
	K_loc = dim_k/num_tasks;
	int my_start, my_end;
	int deficit = dim_k%num_tasks;
	//
	my_start = my_rank*K_loc + MIN(deficit, my_rank) + 1;
	if (my_rank < deficit) K_loc++;
	my_end   = my_start + K_loc - 1;
	//	
	if ((my_end > dim_k - 1) && (my_rank == num_tasks - 1)) my_end = dim_k;
	//
#endif
	//
	int ir;
	int planesize = dim_n*dim_m;
	//
	double* restrict storage1 = (double*)malloc(planesize*(K_loc + 2)*sizeof(double));
	double* restrict storage2 = (double*)malloc(planesize*(K_loc + 2)*sizeof(double));
	//
	memset(storage1, 0., planesize*(K_loc + 2)*sizeof(double));
	memset(storage2, 0., planesize*(K_loc + 2)*sizeof(double));
	//
	if (my_rank == 0)
	{
		printf("3x3 stencil: M=%d N=%d K=%d global, local K = %d, ", dim_n, dim_m, dim_k, K_loc);
		printf("array size = %f MB\n", (double) planesize*sizeof(double)/1024/1024);
	}
	//
	//
	MPI_Barrier(MPI_COMM_WORLD);
	//
	double alloctime = -myseconds();
	//
	//printf("Init array from %d\n", my_start);
	//
	init (storage1, dim_m, dim_n, dim_k, K_loc, my_start, my_end);
	init (storage2, dim_m, dim_n, dim_k, K_loc, my_start, my_end);
#if 0
	for (ir = 0; ir < num_tasks; ++ir) 
	{
		if (my_rank == ir) 
		{
			printf("rank %d: global start = %d, global end = %d, size = %d\n", my_rank, my_start, my_end, K_loc);
			for (int k = 0; k < K_loc + 2; ++k)
			{
				printf("local = %d, global = %d\n", k, my_start + k - 1);
				print(storage1 + k*planesize, dim_m, dim_n);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		//
	}
#endif
	//
	alloctime += myseconds();
	//printf("Allocation time = %f s.\n\n", alloctime);
	int n_iters = 0;
	int nrep    = NREP;
	int ii;
	//
	double* st_read  = storage1;
	double* st_write = storage2;
	//
	double phi;
	double norminf, norml2, gnorm;
	//norminf = maxNorm(st_write, st_read, dim_n*dim_m*K_loc);
	MPI_Barrier(MPI_COMM_WORLD);
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
		// Halo exchange 
		//
# if 1
		int ierr;
		MPI_Status status;
		if (my_rank%2 == 1)
		{
			int tag=666;
			//printf("  %d sendind   plane %d to %d\n", my_rank, 1, my_rank - 1);
			ierr = MPI_Send(st_read  + (        1)*planesize, planesize, MPI_DOUBLE, my_rank - 1, tag, MPI_COMM_WORLD);
			//printf("  %d receiving plane %d from %d\n", my_rank, 0, my_rank - 1);
			ierr = MPI_Recv(st_write + (        0)*planesize, planesize, MPI_DOUBLE, my_rank - 1, tag, MPI_COMM_WORLD, &status);
			if (my_rank < num_tasks - 1)
			{
				int tag=666;
				//printf("**%d sendind   plane %d to %d\n", my_rank, K_loc + 0, my_rank + 1);
				ierr = MPI_Send(st_read  + (K_loc + 0)*planesize, planesize, MPI_DOUBLE, my_rank + 1, tag, MPI_COMM_WORLD);
				//printf("**%d receiving plane %d from %d\n", my_rank, K_loc + 1, my_rank + 1);
				ierr = MPI_Recv(st_write + (K_loc + 1)*planesize, planesize, MPI_DOUBLE, my_rank + 1, tag, MPI_COMM_WORLD, &status);
			}
		}
		else
		{
			if (my_rank > 0)
                        {
                                int tag=666;
                                //printf("  %d receiving plane %d from %d\n", my_rank, 0, my_rank - 1);
                                ierr = MPI_Recv(st_write + (        0)*planesize, planesize, MPI_DOUBLE, my_rank - 1, tag, MPI_COMM_WORLD, &status);
                                //printf("  %d sendind   plane %d to %d\n", my_rank, 1, my_rank - 1);
                                ierr = MPI_Send(st_read  + (        1)*planesize, planesize, MPI_DOUBLE, my_rank - 1, tag, MPI_COMM_WORLD);
                        }
                        if (my_rank < num_tasks - 1)
                        {
                                int tag=666;
                                //printf("**%d receiving plane %d from %d\n", my_rank, K_loc + 1, my_rank + 1);
                                ierr = MPI_Recv(st_write + (K_loc + 1)*planesize, planesize, MPI_DOUBLE, my_rank + 1, tag, MPI_COMM_WORLD, &status);
                                //printf("**%d sendind   plane %d to %d\n", my_rank, K_loc + 0, my_rank + 1);
                                ierr = MPI_Send(st_read  + (K_loc + 0)*planesize, planesize, MPI_DOUBLE, my_rank + 1, tag, MPI_COMM_WORLD);
                        }
		}
#endif
		//
		//
		//
#if 0
		int ir;
		for (ir = 0; ir < num_tasks; ++ir)
		{
			if (my_rank == ir)
			{
				printf("rank %d: global start = %d, global end = %d, size = %d\n", my_rank, my_start, my_end, K_loc);
				for (int k = 0; k < K_loc + 2; ++k)
				{
					printf("local = %d, global = %d\n", k, my_start + k - 1);
					print(storage1 + k*planesize, dim_m, dim_n);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
#endif
		//
		//      
		//
		int b0 = 0, b1 = 0;
		if (my_start ==         1) b0 = 1; 
		if (my_end   == dim_k - 0) b1 = 1; 
		//printf("%d: my_start = %d, my_end = %d, dim_k - 1 = %d, b0 = %d, b1 = %d, K_loc - b1 = %d\n", my_rank, my_start, my_end, dim_k - 1, b0, b1, K_loc - b1);
		//
		// stencil computation
		//
		//if (my_rank == 0) norminf = maxNorm(st_write + planesize, st_read + planesize, planesize*K_loc);
		double ftime = -myseconds();
		unsigned long long c0 = cycles();
		//
		for (ii = 0; ii < nrep; ++ii)
		{
			laplacian(st_read + b0*planesize, st_write + b0*planesize, dim_m, dim_n, K_loc - b1);
		}
#if 0
		if (my_rank == 0) printf("uo ----\n");
                for (ir = 0; ir < num_tasks; ++ir)
                {
                        if (my_rank == ir)
                        {
                                printf("rank %d: global start = %d, global end = %d, size = %d\n", my_rank, my_start, my_end, K_loc);
                                for (int k = 0; k < K_loc + 2; ++k)
                                {
                                        printf("local = %d, global = %d\n", k, my_start + k - 1);
                                        print(st_read + k*planesize, dim_m, dim_n);
                                }
                                printf("\n");
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                }
#endif
#if 0
		if (my_rank == 0) printf("un ----\n");
		for (ir = 0; ir < num_tasks; ++ir)
		{
			if (my_rank == ir)
			{
				printf("rank %d: global start = %d, global end = %d, size = %d\n", my_rank, my_start, my_end, K_loc);
				for (int k = 0; k < K_loc + 2; ++k)
				{
					printf("local = %d, global = %d\n", k, my_start + k - 1);
					print(st_write + k*planesize, dim_m, dim_n);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
#endif
		//
		unsigned long long c1 = cycles();
		unsigned long long cycles = (c1 - c0)/nrep;
		//
		ftime += myseconds();
		ftime /= nrep;
		//
		norminf = maxNorm(st_write + planesize, st_read + planesize, planesize*K_loc);
		//printf("my_rank = %d, maxNorm = %f\n", my_rank, norminf);
				//norml2  = l2Norm (st_write , st_read, dim_n*dim_m*K_loc);
		
		MPI_Allreduce( &norminf, &gnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				//
		double flops = (dim_m - 2)*(dim_n - 2)*(K_loc - 2)*6./1e9;
		if(my_rank == 0) 
		{ 
			printf("iter = %d, %f s. %ld c., linf = %g l2 = %g ", n_iters, ftime, cycles, norminf, norml2);
			printf("%g Gflops, %g c, %g F/c, %g GF/s\n", flops, (double) flops/cycles*1e9, (double) flops/ftime); 
		}
	} while (gnorm > EPSI);
	time += myseconds();
	//
	if (my_rank == 0) printf("\n# iter= %d time= %g residual= %g\n", n_iters, time, norml2); 
	//
	free(storage1);
	free(storage2);
	//
	printf("rank %d\n", my_rank);
	if (my_rank == 0) printf("Shuting down MPI...\n");
#ifdef MPI
	MPI_Finalize();
#endif
	return 0;
}
