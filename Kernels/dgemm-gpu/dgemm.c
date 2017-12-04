#include <stdio.h>       /* standard I/O routines                 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>       /* standard I/O routines                 */
#include <omp.h>

#include <sys/time.h>

#include <cuda.h>
#include <math.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include "utils.h"

#define NNN 1000;

static int NITER = 1;

double myseconds()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int get_acc_energy()
{
	int value;
	char buff[16];

	FILE *fid;

	fid =  fopen("/sys/cray/pm_counters/accel_energy", "r");
	fscanf(fid, "%d %s", &value, buff);
	//printf("%d %s ", value, buff);
	fclose(fid);

	return value;
}


int get_energy()
{
	int value;
	char buff[16];

	FILE *fid;

	fid =  fopen("/sys/cray/pm_counters/energy", "r");
	fscanf(fid, "%d %s", &value, buff);
	//printf("%d %s ", value, buff);
	fclose(fid);

	return value;
}

int main (int argc, char *argv[])
{
	//MPI_Init(&argc, &argv);
	int M, N, K;
	int gpu = 0;
	M = N = K = NNN;
	if ( argc == 2 ) 
	{
		N = atoi(argv[1]);
		M = N;
		K = N;
	}
	else if (argc == 4)
	{
		M = atoi(argv[1]);
		N = atoi(argv[2]);
		K = atoi(argv[3]);
	}
	else if (argc == 5)
	{
                M = atoi(argv[1]);
                N = atoi(argv[2]);
                K = atoi(argv[3]);
		gpu = atoi(argv[4]);
        }
	//
	printf("%dx%dx%d cublas DGEMM...\n", M, N, K);
	//
	//printf("Calling cuda init... ");fflush(stdout);
	//cuInit(0);
	int dev   = 0;
	cudaSetDevice(gpu);
	cudaGetDevice(&dev);
	cublasStatus status; 
	status = cublasInit();
	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, dev);
	printf("GPU %d: 0000:%02x:%02x.0, status: %d\n", dev, props.pciBusID, props.pciDeviceID, status);fflush(stdout);
	//printf("Ncpu = %d, Ngpu = %d\n", Ncpu, Ngpu);
	//pthread_t  threads[NUM_THREADS_GPU + NUM_THREADS_CPU];

	double  alpha =   1.;
	double  beta  =   2.;

	// int N1  = N/NUM_THREADS;
	// int N2  = N - N1;

	int lda    = M;
	int ldb    = K;
	int ldc    = M;

	long int size_A = K*lda;
	long int size_B = N*ldb;
	long int size_C = N*ldc;
	//
	double atime = -myseconds();
	//
	printf("Total CPU memory size: %ld GB\n", (long) (size_A + size_B + size_C)*8/1024/1024/1024);
	//
#ifdef PINNED
	double *A;
	cudaMallocHost((void**) &A, sizeof(double)*size_A);
#else
	double *A  = (double*) malloc(sizeof(double)*size_A);
#endif
	if (A == 0) printf("Could not allocate A.\n");

#ifdef PINNED
	double *B;
	cudaMallocHost((void**) &B, sizeof(double)*size_B);
#else
	double *B  = (double*) malloc(sizeof(double)*size_B);
#endif
	if (B == 0) printf("Could not allocate B.\n");

#ifdef PINNED
	double *Cb;
	cudaMallocHost((void**) &Cb, sizeof(double)*size_C);
#else
	double *Cb = (double*) malloc(sizeof(double)*size_C);
#endif
	if (Cb == 0) printf("Could not allocate Cb (%d GB).\n", size_C);
	atime += myseconds();

	printf("Allocation time = %f s.\n", atime);

	printf("Setting up... "); fflush(stdout);
	double ftime = -myseconds();
	fill(A,  size_A,  0.);
	//eye (B,     ldb,   N );
	fill(B,  size_B,  0.);
	fill(Cb, size_C,  0.);


#ifdef CHECK
	double *Cg = (double*) malloc(sizeof(double)*size_C);
	memcpy(Cg, Cb, size_C*sizeof(double)); 
#endif
	ftime += myseconds();
	printf("Setup time = %f s.\n", ftime);


	//fill(Cg, size_C,  1.);
	//fill(Cb, size_C,  myseconds()*10000);

	int t;

	char transa = 'N';
	char transb = 'N';

	double otime = 0.;

	printf("%d %d %d ", M, N, K);
	fflush(stdout);

	//int energy     = -get_energy();
	//int acc_energy = -get_acc_energy();
#if 1
	printf("calling dgemm...\n"); fflush(stdout);
	double dtime = -myseconds();

	dgemm_buffer(&transa, &transb,
			&M, &N, &K,
			&alpha,
			A, &lda,
			B, &ldb,
			&beta,
			Cb, &ldc);


	dtime += myseconds();
	//
	//energy     += get_energy();
	//acc_energy += get_acc_energy();
	//
	printf("dgemm Gflops = %f (%f s.)\n ",2.*M*N*K/NITER/1e9/dtime, dtime);
	//printf("overlapping: %f %f\n",2.*M*N*K/1.e9/(dtime/(0. + NITER)), dtime/(0. + NITER));
	//printf("node energy = %d J, power consumption = %f W\n", energy, energy/(dtime));
	//printf("acc node energy = %d J, acc power consumption = %f W\n", acc_energy, acc_energy/(dtime));
	fflush(stdout);
#endif

#ifdef CHECK
	//double cputime = -myseconds();	
	dgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, Cg, &ldc);
	//cputime += myseconds();

	//printf("%f %f ",2.*M*N*K/1e9/cputime, cputime);

	//printf("-> %f %f\n", *d_C, *Cb);
	printf("||C - Cg||_max = %f\n", verifyResult(Cb, Cg, M, N));
#endif

#if 0
	free(A);
	free(B);
	free(C);
	free(Cg);
	free(Cb);
#else
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(Cb);
#endif


	cublasShutdown();
	//MPI_Finalize();
	exit(0);

}
