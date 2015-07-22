// author: Gilles Fourestey (CSCS)
#include <stdio.h>       /* standard I/O routines                 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>       /* standard I/O routines                 */
#include <omp.h>

#include <sys/time.h>

#include <cuda.h>
//#include <math.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include "utils.h"

#define NNN 1000;
#define PINNED

double myseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int main (int argc, char *argv[])
{
	int N = NNN;
	if ( argc == 2 ) 
	{
		N = atoi(argv[1]);
	}
	int M = N;
	int K = N;

	printf("%dx%dx%d matrix\n", M, N, K);
	fflush(stdout);
	
	printf("Calling cuInit:\t ");fflush(stdout);
	int ii;
		cublasStatus status; 
	double inittime = -myseconds();
	for (ii = 0; ii < 8; ++ii)
	{
		cudaSetDevice(ii);
		cuInit(0);
		status = cublasInit();
	}
	inittime += myseconds();
	
	printf("%f s.\n", inittime);fflush(stdout);
	//printf("Ncpu = %d, Ngpu = %d\n", Ncpu, Ngpu);
	//pthread_t  threads[NUM_THREADS_GPU + NUM_THREADS_CPU];

	double  alpha =  2.;
	double  beta  =  1.;

	// int N1  = N/NUM_THREADS;
	// int N2  = N - N1;

	int lda    = M;
	int ldb    = K;
	int ldc    = M;

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;


	printf("Mem allocation:\t ");fflush(stdout);
	inittime = -myseconds();
#ifdef PINNED
	double *A;
	cudaHostAlloc((void**) &A, sizeof(double)*size_A, cudaHostAllocPortable);
#else
	double *A  = (double*) malloc(sizeof(double)*size_A);
#endif
	if (A == 0) printf("Could not allocate A.\n");

#ifdef PINNED
	double *B;
	cudaHostAlloc((void**) &B, sizeof(double)*size_B, cudaHostAllocPortable);
#else
	double *B  = (double*) malloc(sizeof(double)*size_B);
#endif
	if (B == 0) printf("Could not allocate B.\n");

#ifdef PINNED
	double *C;
	cudaHostAlloc((void**) &C, sizeof(double)*size_C, cudaHostAllocPortable);
#else
	double *C  = (double*) malloc(sizeof(double)*size_C);
#endif
	if (C == 0) printf("Could not allocate C.\n");

#ifdef PINNED
	double *C_ref;
	cudaHostAlloc((void**) &C_ref, sizeof(double)*size_C, cudaHostAllocPortable);
#else
	double *C_ref = (double*) malloc(sizeof(double)*size_C);
#endif
	if (C_ref == 0) printf("Could not allocate C_ref.\n");

	inittime += myseconds();
        printf("%f s.\n", inittime);fflush(stdout);
	
	printf("Mem setup:\t ");fflush(stdout);
	inittime = -myseconds();
	fill(A    ,  M*K   ,  1.);
	//eye (B,     ldb,   N );
	fill(B    ,  K*N   ,  1.);
	fill(C    ,  size_C,  1.);
	memcpy(C_ref, C, size_C*sizeof(C[0]));
	inittime += myseconds();
        printf("%f s.\n", inittime);fflush(stdout);

	//fill(C_ref,  size_C,  0.);
	//fill0(C,  M,  N);
	//printf("A = %f, %f B = %f, %f\n", *A, *(A + M), *B, *(B + K));

	//memcpy(Cg, C, size_C*sizeof(double)); 
	//memcpy(C_ref, C, size_C*sizeof(double)); 
	//fill(Cg, size_C,  1.);
	//fill(Cb, size_C,  myseconds()*10000);

	int t;

	char transa = 'N';
	char transb = 'N';

	double otime = 0.;



#if 0
	otime -= myseconds();

//	printf("Calling dgemm... ");
	dgemm_cuda(&transa, &transb, 
	//cscs_dgemm(transa, transb, 
		&M, &N, &K, 
		&alpha, 
		A, &lda, 
		B, &ldb, 
		&beta, 
		C, &ldc); 

	otime += myseconds();

	//printf("Overall Gflops = %f\n", 2.*M*N*K/1e6/otime);      	
//	printf("%d: %f %f\n\n", N, 2.*M*N*K/1e9/otime, otime);      	
	finalize();

	printf("%f ",2.*M*N*K/1e9/otime);
	fflush(stdout);
#endif


#if 0
	double mtime = 0.;
    mtime -= myseconds();

	double *d_A , *d_B , *d_C;

	status = cublasAlloc( M*K, sizeof(double), (void**)&d_A );
	if (status) printf("status error %d\n", status);
    status = cublasAlloc( N*K, sizeof(double), (void**)&d_B ) ;
	if (status) printf("status error %d\n", status);
    status = cublasAlloc( M*N, sizeof(double), (void**)&d_C ) ;
	if (status) printf("status error %d\n", status);


    status = cublasSetMatrix( M, K, sizeof(double), A,  lda, d_A, M ) ;
	if (status) printf("status error %d\n", status);
    status = cublasSetMatrix( K, N, sizeof(double), B,  ldb, d_B, K ) ;
	if (status) printf("status error %d\n", status);
    status = cublasSetMatrix( M, N, sizeof(double), Cg, ldc, d_C, M ) ;
	if (status) printf("status error %d\n", status);
	
	double stime = 0.;
	stime -= myseconds();

	
    //magmablas_dgemm(transa, transb,
   	cublasDgemm(transa, transb,
    //cscs_dgemm(transa, transb, 
        M, N, K,
        alpha,
        d_A, M,
        d_B, K,
        beta,
        d_C, M);


	cublasGetMatrix( M, N, sizeof( double ), d_C, M, Cb, ldc ) ;
	stime += myseconds();
    mtime += myseconds();
	//printf("MAGMABLAS %d: %f %f\n\n", N, 2.*M*N*K/1e9/mtime, mtime);

	printf("cublasDgemm: %f %f ",2.*M*N*K/1e9/mtime, 2.*M*N*K/1e9/stime);
	fflush(stdout);
#endif

#if 1
	double dtime = -myseconds();

	my_dgemm(&transa, &transb,
			&M, &N, &K,
			&alpha,
			A, &lda,
			B, &ldb,
			&beta,
			C, &ldc);


	dtime += myseconds();
	printf("%f Gflops (%f s.)\n",2.*M*N*K/1e9/dtime, dtime);
	fflush(stdout);
#endif

#if 1
	if (M < 20000)
	{
		//double cputime = -myseconds();	
		dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &lda, &beta, C_ref, &lda);
		//printf("%f (%f s.) ",2.*M*N*K/1e9/cputime, cputime);

		printf("||C - C_ref||_max = %f\n", verifyResult(C, C_ref, M, N));
	}
#endif
	printf("\n");
#ifndef PINNED
	free(A);
	free(B);
	free(C);
	free(C_ref);
#else
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFreeHost(C_ref);
#endif

	cublasShutdown();
	exit(0);
}
