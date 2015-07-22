#include <stdio.h>
#include <iostream>
#include <cuda.h>
#define BLOCK_DIM 32

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
 
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        printf("%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s\n", file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
 
    }
}

__global__ 
void matrixAddKernel(double *a, double *b, int N)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	//
	if (index < N)
	{
		a[index] += b[index];
	}
}


extern "C"
{
void
matrixAdd(double *host_a, double *client_b, int N, int host_device, int client_device)
{

	int dimBlock = BLOCK_DIM;
	int dimGrid = (int) ceil(N/dimBlock);

	cudaSetDevice(host_device);
	int yes = 0;
	cudaDeviceCanAccessPeer( &yes, host_device, client_device );
	if( yes != 1 ) {
		printf("Cannot access %d from device %d\n", host_device, client_device);
		exit(-1);
	}

	cudaDeviceEnablePeerAccess( client_device, 0 );
	matrixAddKernel<<<dimGrid, dimBlock>>>(host_a, client_b, N);
}


}
