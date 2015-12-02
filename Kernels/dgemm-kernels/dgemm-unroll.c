//author gilles.fourestey@epfl.ch
#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>


#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 240
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 3000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 240
#endif
#else
#define N_BLOCK_SIZE BLOCK_SIZE
#define M_BLOCK_SIZE BLOCK_SIZE
#define K_BLOCK_SIZE BLOCK_SIZE
#endif

#define PAGESIZE 4096;
#define NUMPERPAGE 512 // # of elements to fit a page


#define PREFETCH(A) _mm_prefetch(A, _MM_HINT_NTA)
#define PREFETCH0(A) _mm_prefetch(A, _MM_HINT_T0)
#define PREFETCH1(A) _mm_prefetch(A, _MM_HINT_T1)
#define PREFETCH2(A) _mm_prefetch(A, _MM_HINT_T2)


#define min(a,b) (((a)<(b))?(a):(b))

//double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
//double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE];
//double Cb[M_BLOCK_SIZE*N_BLOCK_SIZE];


#define A_BLOCK
#define B_BLOCK
//#define C_BLOCK



double myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}




void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;

	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
	double Cb[M_BLOCK_SIZE*N_BLOCK_SIZE];

	double copytime    = 0.;
        double computetime = 0.;

	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ // GEMM
		int Kb = min( K_BLOCK_SIZE, K - kb );

		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ // GEPP
			int Mb = min( M_BLOCK_SIZE, M - ib );
			//copytime -= myseconds();
#ifdef A_BLOCK
			for(k = 0; k < Kb; ++k)
				for (i = 0; i < Mb; i = i + 1)
				{
					//PREFETCH1(Ab + (i + 2)*K_BLOCK_SIZE + k);
					//PREFETCH0(Ab + (i + 3)*K_BLOCK_SIZE + k);
					//PREFETCH(Ab + (i + 4)*K_BLOCK_SIZE + k);
					Ab[(i + 0)*K_BLOCK_SIZE + k] = A[(k + kb)*lda + i + ib]; 
				}
#endif
                        //copytime    += myseconds();
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ // GEBP
				int Nb = min( N_BLOCK_SIZE, N - jb );
#ifdef B_BLOCK
                                for (j = 0; j < Nb; j = j + 1)
                                        for (k = 0; k < Kb; k = k + 1){
                                                //PREFETCH(Bb + (j + 4)*K_BLOCK_SIZE + k);
						Bb[j*K_BLOCK_SIZE + k] = B[(j + jb)*ldb + k + kb];
					}
#endif
				//
#ifdef C_BLOCK
                                for (j = 0; j < Nb; j = j + 1)
                                        for (i = 0; i < Mb; i = i + 1){
						//PREFETCH0(Cb + (j + 4)*M_BLOCK_SIZE + i);
                                                //PREFETCH(Cb + (j + 3)*M_BLOCK_SIZE + i);
                                                Cb[j*M_BLOCK_SIZE + i] = C[(j + jb)*ldc + i + ib];
                                        }
#endif
				//computetime -= myseconds();
				for (i = 0; i < Mb - Mb%4; i = i + 4)
				{
					for (j = 0; j < Nb - Nb%2; j = j + 2)
					{
						double c00 = 0.;
						double c01 = 0.;
						double c02 = 0.;
						double c03 = 0.;
						//
						double c10 = 0.;
						double c11 = 0.;
						double c12 = 0.;
						double c13 = 0.;
						//
						double* pA = &Ab[(i + 0)*K_BLOCK_SIZE];
						double* pB = &Bb[(j + 0)*K_BLOCK_SIZE];
						for (k = 0; k < Kb; k = k + 1)
						{
#ifdef B_BLOCK
							double b00 = Bb[(j + 0)*K_BLOCK_SIZE + k];
							double b10 = Bb[(j + 1)*K_BLOCK_SIZE + k];
#else
							double b00 = B[(j + jb + 0)*ldb + k + kb];
							double b10 = B[(j + jb + 1)*ldb + k + kb];
#endif
#ifdef A_BLOCK
							//
							double a00 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a01 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a02 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a03 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							//
							double a10 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a11 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a12 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							double a13 = Ab[(i + 0)*K_BLOCK_SIZE + k];
							//
#endif
							//
							c00 += Ab[(i + 0)*K_BLOCK_SIZE + k]*b00;
							c01 += Ab[(i + 1)*K_BLOCK_SIZE + k]*b00;
							c02 += Ab[(i + 2)*K_BLOCK_SIZE + k]*b00;
							c03 += Ab[(i + 3)*K_BLOCK_SIZE + k]*b00;
							//
							c10 += Ab[(i + 0)*K_BLOCK_SIZE + k]*b10;
							c11 += Ab[(i + 1)*K_BLOCK_SIZE + k]*b10;
							c12 += Ab[(i + 2)*K_BLOCK_SIZE + k]*b10;
							c13 += Ab[(i + 3)*K_BLOCK_SIZE + k]*b10;
							//
						}
#ifdef C_BLOCK
						//
						Cb[(j + 0)*M_BLOCK_SIZE + i + 0] += c00;
						Cb[(j + 0)*M_BLOCK_SIZE + i + 1] += c01;
						Cb[(j + 0)*M_BLOCK_SIZE + i + 2] += c02;
						Cb[(j + 0)*M_BLOCK_SIZE + i + 3] += c03;
						//
						Cb[(j + 1)*M_BLOCK_SIZE + i + 0] += c10;
						Cb[(j + 1)*M_BLOCK_SIZE + i + 1] += c11;
						Cb[(j + 1)*M_BLOCK_SIZE + i + 2] += c12;
						Cb[(j + 1)*M_BLOCK_SIZE + i + 3] += c13;
						//
#else
						C[(j + jb + 0)*ldc + i + ib + 0] += c00;               
						C[(j + jb + 0)*ldc + i + ib + 1] += c01; 
						C[(j + jb + 0)*ldc + i + ib + 2] += c02;
						C[(j + jb + 0)*ldc + i + ib + 3] += c03;
						//
						C[(j + jb + 1)*ldc + i + ib + 0] += c10;
						C[(j + jb + 1)*ldc + i + ib + 1] += c11;
						C[(j + jb + 1)*ldc + i + ib + 2] += c12;
						C[(j + jb + 1)*ldc + i + ib + 3] += c13;
						//
#endif
					}
				}
#ifdef C_BLOCK
				for (j = 0; j < Nb; j = j + 1){
					for (i = 0; i < Mb; i = i + 1){
						C[(j + jb + 0)*ldc + i + ib + 0] = Cb[(j + 0)*M_BLOCK_SIZE + i + 0];
					}
				}
#endif
			}
		}
	}
	//printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}
