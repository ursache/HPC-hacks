//author gilles.fourestey@epfl.ch
#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "oncopy.h"
#include "otcopy.h"
#include "otcopy_8.h"

#ifdef IACA
#include "iacaMarks.h"
#endif

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 200
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 3000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 200
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


//#define STORE128(A, B) _mm_stream_pd(A, B)
#define STORE128(A, B) _mm_store_pd(A, B)

int __attribute__((_stdcall)) dgemm_asm(long int bm,
		long int bn,
		long int bk,
		double alpha,
		double* ba,
		double* bb,
		double* C,
		long int ldc);

double myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void print256(__m256d val)
{
	double a[4] = {0., 0., 0., 0.};
	_mm256_store_pd(&a[0], val);
	printf("%f %f %f %f", a[0], a[1], a[2], a[3]);

}

void print128(__m128d val)
{
        double a[2];
        _mm_store_pd(&a[0], val);
        printf("%f %f", a[0], a[1]);

}


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j;

        double* Ab = _mm_malloc(M_BLOCK_SIZE*K_BLOCK_SIZE*sizeof(double), 64);
        double* Bb = _mm_malloc(K_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double), 64);

	double copytime    = 0.;
        double computetime = 0.;

	//
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
				copytime -= myseconds();
				oncopy(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
				copytime += myseconds();
		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
				copytime -= myseconds();
			otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
				copytime += myseconds();
			//

				computetime -= myseconds();
				double* pA = &Ab[0];
				double* pB = &Bb[0];
				double* pC = &C [jb*ldc + ib];
				double one = 1.;
				//
				dgemm_openblas( (long int) Mb,
						(long int) Nb,
						(long int) Kb,
						one,
						pA,
						pB,
						pC,
						(long int) ldc);
				computetime += myseconds();
			}
		} //
	}
	_mm_free(Ab);
	_mm_free(Bb);
	
	printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}
