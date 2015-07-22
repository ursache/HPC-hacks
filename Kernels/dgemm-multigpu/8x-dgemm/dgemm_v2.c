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

static cudainit = 0;

static cudaEvent_t start, stop;
static int devices[128];

void dgemm_ (char *transa,
		char *transb,
		int *m, int *n, int *k,
		double *alpha, double *A, int *lda,
		double *B, int *ldb,
		double *beta, double *C, int *ldc);


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

double cpu_flops(int N)
{
	if (N < 2000)
		return 160*log(0.008*N)/5.3/14.;
	return 86/14.;
}

double gpu_flops(int N)
{
	return 500*log(0.009*N)/6.1;
}


static __inline
void
block_copy(double *Mdst, double *Msrc, const int N, const int M, int ldaD, int ldaS)
{
    int i, j;
    unsigned cbytes;

    cbytes = M*sizeof(double);

    for (i = 0; i < N; i++)
    {
        //printf("i = %d, size = %dx%d, copying %d bytes from %p to %p\n", i, N, M,  cbytes, Msrc, Mdst); fflush(stdout);
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

		for (i = 0; i < M; ++i)
		{
			*(Mdst_loc++) = (*Msrc_loc++) + beta*(*Mdst_loc);
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

	char* var = getenv("CSCSBLAS_DEBUG");
	int debug = 0;
	if (var != NULL)
		if (strncmp(var, "1", 1) == 0) debug = 1;

	if (debug)
		printf("M=%d, N=%d, K=%d\n", M, N, K);


	if ((M < 64) || (N < 64) || (K < 64)) 
	{
		if (debug) printf("Single thread dgemm_\n");
		dgemm_(&transa, &transb,
				&M, &N, &K,
				&alpha,
				A,                 &lda,
				B , &ldb,
				&beta,
				C , &ldc);		
		return;

	}


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

	//	for (ii = 0; ii < 3; ii++)
	//		printf("%d %d %d\n", mGPU[ii], nGPU[ii], kGPU[ii]);


		//tid = omp_get_thread_num();

	int i, j, k;

	double time;
	time = -mysecond();

	double** d_A = (double**) malloc(sizeof(double*)*nGpuStreams);
	double** d_B = (double**) malloc(sizeof(double*)*nGpuStreams);
	double** d_C = (double**) malloc(sizeof(double*)*nGpuStreams);

	double datatime;

	cublasStatus_t status;

	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 


	int ii;
	for (ii = 0; ii < nGpuStreams; ++ii)
	{
		d_A[ii] = (double*) malloc(sizeof(double*));
		cudaSetDevice(ii);			
		cudaStreamCreate(&streams[ii]);
		cublasCreate    (&handles[ii]);
	}


	// Data transfer host -> device
	int jj;

	for (ii = 0; ii < -nGpuStreams; ii++)
	{
		jj = ii;	
		cudaSetDevice(ii);

		mapping(ii, &i, &j, &k);
		//@printf("device = %d, i = %d, j = %d, k = %d\n", ii, i - 1, j - 1, k - 1);
		int M_n      = mGPU[i] - mGPU[i - 1];
		int N_n      = nGPU[j] - nGPU[j - 1];
		int K_n      = kGPU[k] - kGPU[k - 1];

		//@printf("Set mat: device = %d, M_n = %d, N_n = %d, K_n = %d\n", ii, M_n, N_n, K_n);

		int size_A_n = K_n*M_n;
		int size_B_n = N_n*K_n; 
		int size_C_n = N_n*M_n; 

		datatime -= mysecond();
		status = cudaMalloc((void**) &d_A[ii], size_A_n*sizeof(double));
		if (status) printf("device %d: d_A allocation status error %d\n", ii, status);
		status = cudaMalloc((void**) &d_B[ii], size_B_n*sizeof(double));
		if (status) printf("device %d: d_B allocation status error %d\n", ii, status);
		status = cudaMalloc((void**) &d_C[ii], size_C_n*sizeof(double));
		if (status) printf("device %d: d_C allocation status error %d\n", ii, status);
		datatime += mysecond();

		datatime  = -mysecond();

		status =  cublasSetMatrixAsync(M_n, K_n, sizeof( double ), A + kGPU[k - 1]*lda + mGPU[i - 1], lda, d_A[ii], M_n, streams[ii]);
		if (status) printf("device %d, d_A set mat status error %d, %p\n", devices[ii], status, d_B);

		status = cublasSetMatrixAsync( K_n, N_n, sizeof( double ), B + nGPU[j - 1]*ldb + kGPU[k - 1], ldb, d_B[ii], K_n, streams[ii] ) ;
		if (status) printf("device %d, d_B set mat status error %d, %p\n", devices[ii], status, d_B);

		status = cublasSetMatrixAsync( M_n, N_n, sizeof( double ), C + nGPU[j - 1]*ldc + mGPU[i - 1], ldc, d_C[ii], M_n, streams[ii] ) ;
		if (status) printf("device %d, d_C set mat status error %d, %p\n", devices[ii], status, d_B);

		//@printf("%f %f %f\n\n", *(A + kGPU[k - 1]*lda), *(B + nGPU[j - 1]*ldb), *(C + nGPU[j - 1]*ldc));

		//@printf("\n");
		datatime += mysecond();

		// Looping over the GPU blocks

		datatime = -mysecond();
		double zero = 0.;

		cublasSetStream(handles[ii], streams[ii]);

		//@printf("device = %d, calling dgemm...\n\n", ii);
		cublasDgemm(handles[ii], convertToOp(transa), convertToOp(transb),
				M_n,
				N_n, 
				K_n,
				&alpha,
				d_A[ii],   
				M_n,
				d_B[ii],  
				K_n,
				&zero,
				d_C[ii],
				M_n);

		datatime += mysecond();
	}
	for (ii = 0; ii < nGpuStreams; ii++)
        {
		//{
		int jj = ii;
		cublasSetStream(handles[jj], streams[jj]);
		cudaSetDevice(jj);
		mapping(jj, &i, &j, &k);

		//@printf("\nGet C: device = %d, i = %d, j = %d, k = %d\n", jj, i - 1, j - 1, k - 1);

		int M_n      = mGPU[i] - mGPU[i - 1];
		int N_n      = nGPU[j] - nGPU[j - 1];
		int K_n      = kGPU[k] - kGPU[k - 1];	
		//
		double *C_h;
		cudaHostAlloc((void**) &C_h, sizeof(double)*M_n*N_n, cudaHostAllocPortable);
		//
		cudaHostAlloc((void**) &C_h, sizeof(double)*M_n*N_n, cudaHostAllocPortable);
		C_h = (double*) malloc(M_n*N_n*sizeof(double));
		datatime = -mysecond();
		status = cublasGetMatrix( M_n, N_n, sizeof( double ), d_C[jj], M_n, C_h, M_n/*, streams[jj]*/ );
		datatime += mysecond();

		//block_copy_add(C + nGPU[j - 1]*ldc + mGPU[i - 1], C_h, beta, M_n, N_n, ldc, M_n);

		if (status) printf("device %d: d_C get mat status error %d\n", devices[jj], status);

		printf("C Data transfer rate = %f (%f)\n", (M_n*N_n)*8*1e-9/datatime, datatime);

		cudaFree(C_h);
		cudaFree(d_A[jj]);
		cudaFree(d_B[jj]);
		cudaFree(d_C[jj]);
	}
}
