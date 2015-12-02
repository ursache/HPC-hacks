//author gilles.fourestey@epfl.ch

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include <sys/types.h>
#include <sys/resource.h>

#include <unistd.h>
#include <sys/time.h>

#include "mm_malloc.h"
//#include <mkl.h>

#ifdef PAPI
#include "cscs_papi.h"
#endif

#define _alpha 1.e0
#define _beta  1.e0

#define NN 512
#define NRUNS 1

//#define PAPI

/*
 *   Your function _MUST_ have the following signature:
 */


double mysecond()
{
	struct timeval  tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


static void init(double *A, int M, int N, double v)
{
        int ii, jj;
        for (ii = 0; ii < M; ++ii)
                for (jj= 0; jj < N; ++jj)
                        A[jj*M + ii] = v + jj + 0*(jj*N + ii + 0);
}

static void init_t(double *A, int M, int N, double v)
{
        int ii, jj;
        for (ii = 0; ii < M; ++ii)
                for (jj= 0; jj < N; ++jj)
                        A[jj*M + ii] = ii + 0*(ii + 0);
}


void 
matrix_init (double *A, const int M, const int N, double val)
{
	int i;

	for (i = 0; i < M*N; ++i) 
	{
		A[i] = val*drand48();
		//A[i] = M*val/N;
	}
}

void 
matrix_clear (double *C, const int M, const int N) 
{
	memset (C, 0, M*N*sizeof (double));
}


double
time_dgemm (const int M, const int N, const unsigned K,
                const double alpha, const double *A, const int lda,
                const double *B, const int ldb,
                const double beta, double *C, const unsigned ldc)
{
	double mflops, mflop_s;
	double secs = -1;

	int num_iterations = NRUNS;
	int i;

	double* Ca = (double*) _mm_malloc(N*ldc*sizeof(double), 32);

	double cpu_time = 0;

	double last_clock = mysecond();
	for (i = 0; i < num_iterations; ++i) 
	{
		memcpy(Ca, C, N*ldc*sizeof(double));
		cpu_time -= mysecond();
#ifdef PAPI
		PAPI_START;
#endif
		dgemm (M, N, K, alpha, A, lda, B, ldb, beta, Ca, ldc);
#ifdef PAPI
		PAPI_STOP;
		PAPI_PRINT;
#endif
		cpu_time += mysecond();
	}

	mflops  = 2.0 * num_iterations*M*N*K/1.0e6;
	secs    = cpu_time;
	mflop_s = mflops/secs;


	memcpy(C, Ca, N*ldc*sizeof(double));
#ifdef PAPI
	PAPI_FLUSH;
#endif
	_mm_free(Ca);	
	return mflop_s;
}



double time_dgemm_blas(const int M, const unsigned N, const int K,
		const double alpha, const double *A, const int lda, 
		const double *B, const int ldb,
		const double beta, double *C, const int ldc)
{

	double mflops, mflop_s;
	double secs = -1;

	int num_iterations = NRUNS;
	int i;

	char transa = 'n';
	char transb = 'n';

	double* Ca = (double*) _mm_malloc(N*ldc*sizeof(double), 32);

	double cpu_time = 0;

	for (i = 0; i < num_iterations; ++i)
	{
		memcpy(Ca, C, N*ldc*sizeof(double));
		cpu_time -= mysecond();	
#ifdef PAPI
		PAPI_START;
#endif
		dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, Ca, &ldc);
#ifdef PAPI
		PAPI_STOP;
		PAPI_PRINT;
#endif
		//dgemm (M, N, K, alpha, A, lda, B, ldb, beta, Ca, ldc);
		cpu_time += mysecond();
	}

	mflops  = 2.0*num_iterations*M*N*K/1.0e6;
	secs    = cpu_time;
	mflop_s = mflops/secs;

	memcpy(C, Ca, N*ldc*sizeof(double));
#ifdef PAPI
	PAPI_FLUSH;
#endif
	_mm_free(Ca);

	return mflop_s;
}

	int
main (int argc, char *argv[])
{
	int sz_i;
	double mflop_s, mflop_b;

	int M, N, K;

	if ( argc == 4 )
	{
		M    = atoi(argv[1]);
		N    = atoi(argv[2]);
		K    = atoi(argv[3]);
	}
	else if (argc == 2)
	{
		M    = atoi(argv[1]);
		N = M;
		K = M;
	}
	else
	{
		M = NN;
		N = NN;
		K = NN;
	}

	int lda = M;
	int ldb = K;
	int ldc = M;

	double* A  = (double*) _mm_malloc(M*K*sizeof(double), 32);
	double* B  = (double*) _mm_malloc(K*N*sizeof(double), 32);
	double* C  = (double*) _mm_malloc(M*N*sizeof(double), 32);
	double* Cb = (double*) _mm_malloc(M*N*sizeof(double), 32);

#if 1
	matrix_init(A,  M, K, 1.);
	//init_t(A,  M, K, 1.);
	matrix_init(B,  K, N, 2.);
	//init_t(B,  K, N, 1.);
	//matrix_clear(C);
	matrix_init(C,  M, N, 0.);
	memcpy(Cb, C, M*N*sizeof(double));
	//matrix_init(Cb, M, N, 0.);
#endif

	//const int M = test_sizes[sz_i];
	printf("Size: %u %u %u \t", M, N, K); fflush(stdout);

	mflop_s = time_dgemm     (M, N, K, _alpha, A, lda, B, ldb, _beta, C , ldc);    
	printf ("Gflop/s: %g ", mflop_s/1000.); fflush(stdout);

#if 0
	mflop_b = time_dgemm_blas(M, N, K, _alpha, A, lda, B, ldb, _beta, Cb, ldc);
	printf ("blas Gflops: %g\n", mflop_b/1000);

#if 1
	int ii, jj;
	for (jj = 0; jj < N; ++jj)
	{	
		for (ii = 0; ii < M; ++ii)
		{
			if (abs(C[jj*ldc + ii] - Cb[jj*ldc + ii]) != 0)
				printf("i = %d, j = %d, C = %f, should be %f\n", ii, jj, C[jj*ldc + ii], Cb[jj*ldc + ii]);
		}
	} 
#endif
#endif
	printf("\n");
	_mm_free(A);
	_mm_free(B);
	_mm_free(C);
	_mm_free(Cb);


	return 0;
}
