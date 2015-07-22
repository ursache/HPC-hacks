// author: Gilles Fourestey (CSCS)
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <sys/time.h>
//#include <cblas.h>

#include <cuda.h>
#include <math.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>


#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))
#define NSTREAMS 128

static cudaStream_t    streams    [NSTREAMS];
static cublasHandle_t  handles    [NSTREAMS];
static cudaEvent_t     eventsStart[NSTREAMS];
static cudaEvent_t     eventsStop [NSTREAMS];

static cublasHandle_t  handle;

static cudaEvent_t start, stop;
static int devices[128];


void dgemm_ (char *transa,
		char *transb,
		int *m, int *n, int *k,
		double *alpha, double *A, int *lda,
		double *B, int *ldb,
		double *beta, double *C, int *ldc);


void matrixAdd(double*, double*, int, int, int);


#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        printf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ) );

    }
}




static __inline__ cublasOperation_t convertToOp( char trans )
{
    switch(trans) {
        case 'N':
        case 'n':
            return CUBLAS_OP_N;
        case 't':
        case 'T':
            return CUBLAS_OP_T;
        case 'C':
        case 'c':
            return CUBLAS_OP_C;
        default:
            return CUBLAS_OP_N;
    }

}


void mapping(int tid, int* i, int* j, int* k)
{
	switch(tid)
	{
                case 0:
                        *i = 1; *k = 1; *j = 1; break;
                case 1:
                        *i = 1; *k = 2; *j = 1; break;
                case 2:
                        *i = 1; *k = 1; *j = 2; break;
                case 3:
                        *i = 1; *k = 2; *j = 2; break;
                case 4:
                        *i = 2; *k = 1; *j = 1; break;
                case 5:
                        *i = 2; *k = 2; *j = 1; break;
                case 6:
                        *i = 2; *k = 1; *j = 2; break;
                case 7:
                        *i = 2; *k = 2; *j = 2; break;
		default:
			printf("mapping problem: tid = %d!\n", tid);
	}	
	//printf("device = %d, i = %d, j = %d, k = %d\n", tid, *i, *j, *k);
} 



void
scale(double* Mdst, int size, double beta)
{
	if (beta == 1) return;
	int ii;
	double* Mdst_loc = Mdst;
#pragma omp parallel for private(ii)
	for (ii = 0; ii < size; ++ii)
	{
		Mdst_loc[0] = beta*Mdst_loc[0];
		++Mdst_loc;
	}
}


//static __inline 
void
block_copy(double *Mdst, double *Msrc, const int N, const int M, int ldaD, int ldaS)
{
	int i, j;
	unsigned cbytes;

	cbytes = M*sizeof(double);

	for (i = 0; i < N; i++)
	{
		/* Copy the relevent data bytes */
		memcpy(Mdst, Msrc, cbytes);
		Mdst += ldaD;
		Msrc += ldaS;
	}
}

void
block_copy_add(double *Mdst, double *Msrc, double beta, const int N, const int M, int ldaD, int ldaS)
{
	int j, i;

	for (j = 0; j < N; j++)
	{
		int     count    = M;
		double* Msrc_loc = Msrc;
		double* Mdst_loc = Mdst;
#pragma omp parallel for private(i)
		for (i = 0; i < M; ++i)
		{
			Mdst[i] = Msrc[i] + beta*Mdst[i];
			//Mdst_loc++;
			//Msrc_loc++;
		}

		Mdst += ldaD;
		Msrc += ldaS;
	}
}



double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void my_dgemm(const char*  _transa,
		const char*   _transb,
		const int*    _M,
		const int*    _N,
		const int*    _K,
		const double* _alpha,
		const double* _A,
		const int*    _lda,
		const double* _B,
		const int*    _ldb,
		const double* _beta,
		double*       _C,
		const int*    _ldc)
{



	const char    transa = *_transa;
	const char    transb = *_transb;
	const int     M      = *_M;
	const int     N      = *_N;
	const int     K      = *_K;
	const double  alpha  = *_alpha;
	const double* A      = _A;
	const int     lda    = *_lda;
	const double* B      = _B;
	const int     ldb    = *_ldb;
	const double  beta   = *_beta;
	double*       C      = _C;
	const int     ldc    = *_ldc;	

	//@             printf("Stating ...\n" );
	//

	if ((K==0) || (N == 0) || (M == 0)) return;

	int nGpuStreams = 8;

	int mGPU[3];
	mGPU[0] = 0;
	mGPU[1] = M/2;
	mGPU[2] = M;

	int nGPU[3];
	nGPU[0] = 0 ;
	nGPU[1] = N/2;
	nGPU[2] = N;

	int kGPU[3];
	kGPU[0] = 0;
	kGPU[1] = K/2;
	kGPU[2] = K;

	int i, j, k;

	double time;
	time = -mysecond();
	       
        cublasStatus_t status;

	double** d_C = (double**) malloc(sizeof(double*)*nGpuStreams);

	int ii;
	for (ii = 0; ii < nGpuStreams; ++ii)
	{
		cudaSetDevice(ii);			
		cudaStreamCreate(&streams[ii]);
		cublasCreate    (&handles[ii]);
	}
	//	
	int size_C = N*M/nGpuStreams;
	//
	for (ii = 0; ii < nGpuStreams; ++ii)
        {
		cudaSetDevice(ii);
		status = cudaMalloc((void**) &d_C[ii], size_C*sizeof(double));
	}	
	//	
	double* C_h = (double*) malloc(sizeof(double)*M*N);
	status = cudaHostAlloc((void**) &C_h, sizeof(double)*M*N, cudaHostAllocPortable);
	if (status != 0) printf("GPU %d: error status = %d\n", ii, status);
	getLastCudaError("C_h allocation");
	//
	printf("H2D xsfer rate\t\t");fflush(stdout);
	//
	double datatime = -mysecond();
	for (ii = 0; ii < nGpuStreams; ++ii)
	{
		//
		cudaSetDevice(ii);	
		//
		status = cublasSetMatrixAsync( M, N/nGpuStreams, sizeof( double ), C + size_C*ii, ldc, d_C[ii], M, streams[ii]) ;
		if (status != 0) printf("GPU %d: error status = %d\n", ii, status);
		getLastCudaError("H2D transfer");
		//
	}
	cudaDeviceSynchronize();
	datatime += mysecond();
	//	
	printf("%f GB/s(%f)\n", M*N*8./1024/1024/1024/datatime, datatime);
	//
	printf("D2H xsfer rate\t\t");fflush(stdout);
	//
	datatime = -mysecond();
        for (ii = 0; ii < nGpuStreams; ++ii)
        {
		//
		cudaSetDevice(ii);
		//
		status = cublasSetMatrixAsync( M, N/nGpuStreams, sizeof( double ), d_C[ii], M, C_h + size_C*ii, ldc, streams[ii]) ;
		if (status != 0) printf("GPU %d: error status = %d\n", ii, status);
		getLastCudaError("H2D transfer");

		//
	}
	//
	cudaDeviceSynchronize();
	datatime += mysecond();
	printf("%f GB/s(%f)\n", M*N*8./1024/1024/1024/datatime, datatime); fflush(stdout);

	//daxpy_(N*M, -1., C_h, 1, C, 1); 
	//printf("Norm of the difference = %f\n", dnrm2(N*M, C_h, 1));
	for (ii = 0; ii < M*N; ++ii)
	{
		if (fabs(C_h[ii] - C[ii]) > 1.e-5)
			printf("%d = %f %f\n", ii, C_h[ii], C[ii]); 
			exit(-1);
	}

	printf("done\n");fflush(stdout);


	
}
