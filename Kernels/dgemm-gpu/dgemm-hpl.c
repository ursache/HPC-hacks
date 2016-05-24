#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <sys/time.h>
//#include <cblas.h>

#include <cuda.h>
#include <math.h>
#include <cublas.h>
//#include <cublas_v2.h>
#include <cuda_runtime.h>
//#include <nvml.h>

//#include "complex.h"
//#include "timer.h"
//#include <magma.h>

#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))

#define granularity 1000


#define CHECK_CUDART(x) do { \
	cudaError_t res = (x); \
	if(res != cudaSuccess) { \
		fprintf(stderr, "%d : %s : CUDART: %s = %d (%s) at (%s:%d)\n", rank, host_name, #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
		exit(1); \
	} \
} while(0)

#define CHECK_CUBLAS do { \
	cublasStatus_t cublasStatus = cublasGetError(); \
	if(cublasStatus != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "%d : %s : CUBLAS: %d at (%s:%d)\n", rank, host_name, cublasStatus,__FILE__,__LINE__); \
        exit(1); \
    } \
} while(0)

	//cublasStatus_t cublasStatus = cublasCheckError(); \

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)>(b))?(a):(b))


#define NSTREAMS 3

static int NITER    = 1;

static cudaStream_t    streams    [NSTREAMS];
static cublasHandle_t  handles    [NSTREAMS];
static cudaEvent_t     eventsStart[NSTREAMS];
static cudaEvent_t     eventsStop [NSTREAMS];

static cublasHandle_t  handle;

//static double*         d_B        [NSTREAMS];
//static double*         d_C        [NSTREAMS];

static cudaEvent_t start, stop;

#if 0
void dgemm_ (char *transa,
		char *transb,
		int *m, int *n, int *k,
		double *alpha, double *A, int *lda,
		double *B, int *ldb,
		double *beta, double *C, int *ldc);
#endif
int get_power()
{
        int value;
        char buff[16];

        FILE *fid;

        fid =  fopen("/sys/cray/pm_counters/accel_power", "r");
        fscanf(fid, "%d %s", &value, buff);
        //printf("%d ", value);
        fclose(fid);

        return value;
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

double mysecond()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int dgemm_buffer(const char*  _transa,
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

	printf("%d %d %d %d %d %d %d %d: ", transa, transb, M, N, K, lda, ldb, ldc);

	if ((M == 0) || (N == 0) || (K == 0))
	{
		//printf("\n"); 
		return;
	}
//	if ((N < 896) || (K < 896) || (M < 896))
//	{
//		//printf("cpu dgemm_ ");
//		dgemm_(_transa, _transb, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
//		return;
//	}
	int rank, nprocs, local_rank, local_procs;
	char host_name[20]="local";

	//printf("Stating ...\n" );
	//cudaStream_t stream[NSTREAMS];


	char* var = getenv("CSCSBLAS_USEGPU");

	var = getenv("CSCSBLAS_DEBUG");
	int debug = 0;
	if (var != NULL)
		if (strncmp(var, "1", 1) == 0) debug = 1;

	int numDevice = 0;


	// NVML
#if 0
	nvmlInit();

	nvmlDevice_t device;
	CHECK_CUDART(nvmlDeviceGetHandleByIndex(numDevice, &device));

	char name[NVML_DEVICE_NAME_BUFFER_SIZE];

	CHECK_CUDART(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
	//printf("%s, ", name);

	nvmlEnableState_t mode;
	CHECK_CUDART(nvmlDeviceGetPowerManagementMode(device, &mode));
#endif
	//printf("mode = %d\n", mode);

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;

	int t;
	int ii;

	int done = 0;

	int jj;
	//
	double* d_A;
	double* d_B[NSTREAMS];
	double* d_C[NSTREAMS];
	//
	double time;
	double datatime;
	//
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//
	// Data transfer host -> device
	//printf("Mem allocation...\n");
	//
	int buffer_size = NSTREAMS*(K*K + M*K);  
	printf("Total GPU memory used: %lld MB\n", (sizeof(double)*buffer_size)>>20 );
	//
	CHECK_CUDART(cudaMalloc((void**) &d_A, K*M*sizeof(double)));
	//				cudaEventRecord(start, 0);
	for(ii = 0; ii < NSTREAMS; ++ii)
	{
		CHECK_CUDART(cudaMalloc((void**) &d_C[ii], M*K*sizeof(double)));
		CHECK_CUDART(cudaMalloc((void**) &d_B[ii], K*K*sizeof(double)));
		CHECK_CUDART(cudaStreamCreate(&streams[ii]));
	}
	//printf("Data allocation time = %f\n", datatime);
	//
	for (jj = 0; jj < NITER; ++jj)	
	{
		int ib = 0;

		datatime = 0.;
		// Looping over the GPU blocks
		int counter = 0;
		int nq0     = 0;
		int nn;
		time = -mysecond();
		CHECK_CUDART(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
		while( (nn = N - nq0) > 0 )
		{
			nn = imin( nn, K);
			//printf("nn = %d\n", nn);
			//
			//Launching the computation on the host ...
			//
			CHECK_CUDART(cublasSetKernelStream(streams[counter%NSTREAMS]));
			//
			CHECK_CUDART( cudaMemcpyAsync( d_B[counter%NSTREAMS], B + nq0*K, nn*K*sizeof(double), cudaMemcpyHostToDevice, streams[counter%NSTREAMS] ));
			CHECK_CUDART( cudaMemcpyAsync( d_C[counter%NSTREAMS], C + nq0*M, nn*M*sizeof(double), cudaMemcpyHostToDevice, streams[counter%NSTREAMS] ) );
			//double ttime = -mysecond();
			cublasDgemm(transa, transb,
					M,
					nn, 
					K,
					alpha,
					d_A,    //lda
					lda,
					d_B[counter%NSTREAMS],  //ldb,
					K,
					beta,
					d_C[counter%NSTREAMS],  //ldc);
					M);
			//printf("M N K = %d %d %d  nn nq0 = %d %d\n", M, N, K, nn, nq0);
			CHECK_CUBLAS;
			//
			CHECK_CUDART( cudaMemcpyAsync( C + nq0*M, d_C[counter%NSTREAMS], nn*M*sizeof(double), cudaMemcpyDeviceToHost, streams[counter%NSTREAMS] ) );
			nq0 += nn;
			counter++;
		}
		cudaThreadSynchronize();
		time += mysecond();
		//printf("%f Gflops, %f s.\n", 2.*M*N*K/1.e9/time, time);

	}
	cudaFree(d_A);
	for (ii = 0; ii < NSTREAMS; ++ii)
	{
		cudaFree(d_B[ii]);	
		cudaFree(d_C[ii]);	
	}
	done = 1;
}
