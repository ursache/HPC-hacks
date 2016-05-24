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
#include <nvml.h>

//#include "complex.h"
//#include "timer.h"
//#include <magma.h>

#define ERRCODE(e) (-(__LINE__ * 1000 + (e)))


#define CHECK_CUDART(x) do { \
    cudaError_t res = (x); \
    if(res != cudaSuccess) { \
        fprintf(stderr, "%d : %s : CUDART: %s = %d (%s) at (%s:%d)\n", rank, host_name, #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(x) do { \
    cublasStatus_t cublasStatus = (x); \
    if(cublasStatus != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "%d : %s : CUBLAS: %s = %d at (%s:%d)\n", rank, host_name, #x, cublasStatus,__FILE__,__LINE__); \
        exit(1); \
    } \
} while(0)

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)>(b))?(a):(b))

#define KBB 896
//#define KBB 4

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

void dgemm_ (char *transa,
		char *transb,
		int *m, int *n, int *k,
		double *alpha, double *A, int *lda,
		double *B, int *ldb,
		double *beta, double *C, int *ldc);



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
block_copy(double *Mdst, const double *Msrc, const int M, const int N, int ldaD, int ldaS)
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


int* dgemm_buffer(const char*  _transa,
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

	int rank, nprocs, local_rank, local_procs;
	char host_name[20]="local";

	//cudaStream_t stream[NSTREAMS];


	char* var = getenv("CSCSBLAS_USEGPU");
	int usegpu = 1;
	if (var != NULL)
	{
		if (strncmp(var, "1", 1) == 0) usegpu = 1;
	}

	var = getenv("CSCSBLAS_DEBUG");
	int debug = 0;
	if (var != NULL)
		if (strncmp(var, "1", 1) == 0) debug = 1;

	int nGpuStreams = 4;
	var = getenv("CSCSBLAS_GPUSTREAMS");
	if (var != NULL)
		nGpuStreams = atoi(var);

	// NVML
#ifdef NVML
	int numDevice;
	nvmlInit();

	nvmlDevice_t device;
	CHECK_CUDART(nvmlDeviceGetHandleByIndex(numDevice, &device));

	char name[NVML_DEVICE_NAME_BUFFER_SIZE];

	CHECK_CUDART(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
	printf("%s, ", name);

	nvmlEnableState_t mode;
	CHECK_CUDART(nvmlDeviceGetPowerManagementMode(device, &mode));
	printf("NVML mode = %d\n", mode);
#endif

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;

	int t;
	int ii;

	int done = 0;

	double time;
	int jj;
	double* d_A;
	double* d_B[NSTREAMS];
	double* d_C[NSTREAMS];
	//
	double datatime;
	//
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Data transfer host -> device
	printf("Mem allocation...\n");
	//
	double* h_B;
	int KB = imin(K, KBB);
	//
	CHECK_CUDART(cudaMalloc    ((void**) &d_A, M*KB*sizeof(double)));
	//CHECK_CUBLAS(cublasAlloc (M*KB, sizeof(*d_A), (void**)&d_A));
	CHECK_CUDART(cudaMallocHost((void**) &h_B, sizeof(double)*KB*N));
	//				cudaEventRecord(start, 0);
	for(ii = 0; ii < NSTREAMS; ++ii)
	{
		CHECK_CUDART(cudaMalloc((void**) &d_C[ii], M*KB*sizeof(double)));
		CHECK_CUDART(cudaMalloc((void**) &d_B[ii], KB*K*sizeof(double)));
		CHECK_CUDART(cudaStreamCreate(&streams[ii]));
		//
		//CHECK_CUDART(cudaMallocHost((void**) &h_B[ii], sizeof(double)*KB*KB));
	}
	//printf("Data allocation time = %f\n", datatime);
	int buffer_size = NSTREAMS*(M*KB + KB*K) + M*KB;  
	printf("# memory used: %lld MB\n", (sizeof(double)*buffer_size)>>20 );
	datatime = -myseconds();	
	CHECK_CUBLAS(cublasSetMatrix(K, KB, sizeof(double), (void*) B, K, (void*) &d_B[0], K));
	datatime += myseconds();
	printf("Transfer rate = %f %f\n", M*KB/datatime/1024/1024/1024, datatime);
	//
	for (jj = 0; jj < NITER; ++jj)	
	{
		int nn, kk;
		int kb = 0;
		int kcounter = 0;
		//
		time = -mysecond();
		//
		while( (kk = K - kb) > 0 )
		{ 
			kk = imin( kk, KB);
			// Looping over the GPU blocks
			int counter = 0;
			int nb      = 0;
			//
			//CHECK_CUDART(cudaMemcpy(d_A, A + kb*M, M*kk*sizeof(double), cudaMemcpyHostToDevice));
			//CHECK_CUBLAS(cublasSetMatrixAsync(M, kk, sizeof(double), A + kb*lda, lda, d_A, M, streams[0]));
			datatime = -myseconds();	
			CHECK_CUBLAS(cublasSetMatrix(M, kk, sizeof(double), (void*) A + kb*lda, lda, (void*) d_A, M));
			//cudaThreadSynchronize();
			datatime += myseconds();
			printf("Transfer rate = %f %f\n", M*kk/datatime/1024/1024/1024, datatime);
			//printf("Copying from kk = %d, KB = %d  to C, N = %d, ldb = %d\n", kk, KB, N, ldb);  
			//block_copy(double *Mdst, const double *Msrc, const int M, const int N, int ldaD, int ldaS)
			//block_copy(h_B, B + kb, kk, N, KB, ldb);
			//int ik, jk;
			//for (ik = 0; ik < N; ++ik)
			//	for (jk = 0; jk < K; ++jk)
			//		printf("B= %f, h_B = %f\n", *(B + jk + ik*K), *(h_B + jk + ik*KB)); 
			while( (nn = N - nb) > 0 )
			{
				nn = imin( nn, KB);
				printf("kk = %d, kb = %d, nn = %d, nb = %d, kcounter > 0 = %d\n", kk, kb, nn, nb, !(kcounter > 0)); fflush(stdout);
				//
				//Launching the computation on the host ...
				//
				cublasSetKernelStream(streams[counter%NSTREAMS]);
				//
				CHECK_CUDART( cudaMemcpyAsync( d_C[counter%NSTREAMS], C + nb*ldc, nn*M*sizeof(double), cudaMemcpyHostToDevice, streams[counter%NSTREAMS] ) );
				//
				//block_copy(h_B[counter%NSTREAMS], B + nb*K, KB, nn, KB, K);
				//
				CHECK_CUDART( cudaMemcpyAsync( d_B[counter%NSTREAMS], h_B + nb*KB, nn*KB*sizeof(double), cudaMemcpyHostToDevice, streams[counter%NSTREAMS] ) );
				//
				//CHECK_CUDART( cudaMemcpyAsync( d_B[counter%NSTREAMS], B + nb*ldb, nn*K*sizeof(double), cudaMemcpyHostToDevice, streams[counter%NSTREAMS] ) );
				//
				//double ttime = -mysecond();
				//
				double mybeta = (kcounter == 0) ? beta : 1.;
				cublasDgemm(transa, transb,
						M,
						nn, //Ngpu
						kk,
						alpha,
						d_A,    //lda
						M,
						d_B[counter%NSTREAMS],  //ldb,
						KB,
						mybeta,
						d_C[counter%NSTREAMS],  //ldc);
						M);
				//if (error != 0) printf("cublasDgemm error = %d\n", error);
				//
				CHECK_CUDART( cudaMemcpyAsync( C + nb*ldc, d_C[counter%NSTREAMS], nn*M*sizeof(double), cudaMemcpyDeviceToHost, streams[counter%NSTREAMS] ) );
				//cudaDeviceSynchronize();
				//cudaThreadSynchronize();
				//cudaStreamSynchronize(streams[counter%NSTREAMS]);
				nb += nn;
				counter++;
			}
			kb += kk; 
			kcounter++;
			//cudaThreadSynchronize();
			//cudaDeviceSynchronize();
		}

	}
	time += mysecond();
	printf("%f Gflops, %f s.\n", 2.*M*N*K/1.e9/time, time);
	cudaFree(d_A);
	for (ii = 0; ii < NSTREAMS; ++ii)
	{
		CHECK_CUDART(cudaFree(d_B[ii]));	
		CHECK_CUDART(cudaFree(d_C[ii]));	
	}
	CHECK_CUDART(cudaFreeHost(h_B));
}
