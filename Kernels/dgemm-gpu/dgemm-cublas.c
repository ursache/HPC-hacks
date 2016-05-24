#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <sys/time.h>
//#include <cblas.h>

#define USEGPU

#ifdef USEGPU
#include <cuda.h>
#include <math.h>
#include <cublas.h>
//#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>
#endif

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

#define CHECK_CUBLAS(x) do { \
    cublasStatus_t cublasStatus = (x); \
    if(cublasStatus != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "%d : %s : CUBLAS: %s = %d at (%s:%d)\n", rank, host_name, #x, cublasStatus,__FILE__,__LINE__); \
        exit(1); \
    } \
} while(0)

#define imin(a,b) (((a)<(b))?(a):(b))
#define imax(a,b) (((a)>(b))?(a):(b))


static int NITER    = 1;

static cudaEvent_t start, stop;

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
	switch( trans){
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

	printf("Stating ...\n" );
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

	int numDevice = 0;


#ifdef  NVML
	nvmlInit();

	nvmlDevice_t device;
	CHECK_CUDART(nvmlDeviceGetHandleByIndex(numDevice, &device));

	char name[NVML_DEVICE_NAME_BUFFER_SIZE];

	CHECK_CUDART(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
	printf("%s, ", name);

	nvmlEnableState_t mode;
	CHECK_CUDART(nvmlDeviceGetPowerManagementMode(device, &mode));
	printf("mode = %d\n", mode);
#endif

	int size_A = K*lda;
	int size_B = N*ldb;
	int size_C = N*ldc;

	int t;
	int ii;

	int done = 0;

	double time, ttime, otime;
	int jj;
	double* d_A;
	double* d_B;
	double* d_C;
	//
	double datatime;
	//
	int nGPU[nGpuStreams + 1];
	//
	nGPU[0]           = 0;
	nGPU[nGpuStreams] = N;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Data transfer host -> device
	//
	double* h_B;
	//
	CHECK_CUDART(cudaMalloc((void**) &d_A, M*K*sizeof(double)));
	//				cudaEventRecord(start, 0);
	CHECK_CUDART(cudaMalloc((void**) &d_C, M*N*sizeof(double)));
	CHECK_CUDART(cudaMalloc((void**) &d_B, K*N*sizeof(double)));
		//
	//printf("Data allocation time = %f\n", datatime);
	int buffer_size = K*N + M*N + M*K;  
	printf("total GPU memory used: %lld MB\n", (sizeof(double)*buffer_size)>>20 );
	//
	otime -= mysecond();
	for (jj = 0; jj < NITER; ++jj)	
	{


		
		ttime -= mysecond();
		//CHECK_CUDART(cudaMemcpy(d_A, A, M*K*sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasSetMatrix(M, K, sizeof(double), (void*) A, lda, d_A, M));
		//
		//CHECK_CUDART(cudaMemcpy(d_C, C, N*M*sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(double), (void*) C, ldc, d_C, M));
		//
		//CHECK_CUDART(cudaMemcpy(d_B, B, N*K*sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUBLAS(cublasSetMatrix(K, N, sizeof(double), (void*) B, ldb, d_B, K));
		ttime += mysecond();
		//
		//
		time = -mysecond();
		cublasDgemm(transa, transb,
				M,
				N, //Ngpu
				K,
				alpha,
				d_A,    //lda
				M,
				d_B,  //ldb,
				K,
				beta,
				d_C,  //ldc);
			M);
		cudaDeviceSynchronize();
		time += mysecond();
		//
		CHECK_CUDART( cudaMemcpyAsync( C, d_C, N*M*sizeof(double), cudaMemcpyDeviceToHost, 0 ) );
	}
	//
	cudaThreadSynchronize();
	otime += mysecond();
	//
	printf("xfer: %f GB/s (%f s), kernel: %f Gflops (%f s) , full: %f Gflops (%f s)\n", (sizeof(double)*buffer_size>>20)/ttime/NITER, ttime, 2.*M*N*K/1.e9/time/NITER, time/NITER, 2.*M*N*K/1.e9/otime/NITER, otime/NITER);
	cudaFree(d_A);
	cudaFree(d_B);	
	cudaFree(d_C);	
}
