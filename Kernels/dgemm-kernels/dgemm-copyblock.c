//author gilles.fourestey@epfl.ch
//#define BLOCK_SIZE 128
#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif


#if !defined(BLOCK_SIZE)
#warning "local definition of blocks"
#define M_BLOCK_SIZE 240
#define N_BLOCK_SIZE 3000
#define K_BLOCK_SIZE 240
#else
#define N_BLOCK_SIZE BLOCK_SIZE
#define M_BLOCK_SIZE BLOCK_SIZE
#define K_BLOCK_SIZE BLOCK_SIZE
#endif

#define A_BLOCK
#define B_BLOCK
//#define C_BLOCK

#define min(a,b) (((a)<(b))?(a):(b))

#define PREFETCH(A) _mm_prefetch(A, _MM_HINT_T0)

#include <sys/time.h>


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i , j, k;
	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
	double Cb[M_BLOCK_SIZE*N_BLOCK_SIZE];
        double copytime    = 0.;
        double computetime = 0.;

	// Blocking
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){int Kb = min( K_BLOCK_SIZE, K - kb );
		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){int Mb = min( M_BLOCK_SIZE, M - ib );
			// copy/transpose
			for(k = 0; k < Kb; ++k)
				for (i = 0; i < Mb; ++i)
				{
					Ab[i*K_BLOCK_SIZE + k] = A[(k + kb)*lda + (i + ib)];
				}
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){int Nb = min( N_BLOCK_SIZE, N - jb );
#ifdef B_BLOCK
				for (j = 0; j < Nb; j = j + 1)
					for (k = 0; k < Kb - Kb%1; k = k + 1){
						Bb[j*K_BLOCK_SIZE + k + 0] = B[(jb + j)*ldb + k + kb + 0];
					}
#endif
				//
#ifdef C_BLOCK
				for (j = 0; j < Nb; j = j + 1)
					for (i = 0; i < Mb - Mb%1; i = i + 1){
						Cb[j*M_BLOCK_SIZE + i + 0] = C[(jb + j)*ldc + i + ib + 0];
					}
#endif
				// DGEMM kernel
				for (i = 0; i < Mb; i = i + 1)
					for (j = 0; j < Nb; j = j + 1)
					{
						double c11 = 0.;
						for (k = 0; k < Kb; k = k + 1)
						{
							c11 += Ab[(i + 0)*K_BLOCK_SIZE + k]*
#ifdef B_BLOCK
							B[(j + jb)*ldb + (k + kb)];
#endif
						}
#ifdef C_BLOCK
						Cb[j*M_BLOCK_SIZE + i] += alpha*c11;
#else
						C[(j + jb)*ldc + (i + ib)] += alpha*c11;
#endif
					}
#ifdef C_BLOCK
				for (j = 0; j < Nb; j = j + 1)
					for (i = 0; i < Mb; ++i){
						C[(j + jb)*ldc + (i + ib)] = Cb[j*M_BLOCK_SIZE + i];
					}	
#endif
			}
		}
	}
}
