//author gilles.fourestey@epfl.ch

#include <immintrin.h>

#define BLOCK_SIZE 64

#define min(a,b) (((a)<(b))?(a):(b))

#define PREFETCH(A)  _mm_prefetch((void*) A, _MM_HINT_NTA)
#define PREFETCH0(A) _mm_prefetch((void*) A, _MM_HINT_T0)
#define PREFETCH1(A) _mm_prefetch((void*) A, _MM_HINT_T1)
#define PREFETCH2(A) _mm_prefetch((void*) A, _MM_HINT_T2)

#define CLS 64
#define SM ( CLS / sizeof (double))


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;
	int ii, jj, kk;
	//
	double a, b ,cij;
	//
	double *pA, *pB, *pC;
	//
	for (ib = 0; ib < M; ib += SM)
		for (jb = 0; jb < N; jb += SM)
			for (kb = 0; kb < K; kb += SM)
			{
				//
                                int Kb = min( SM, K - kb );
                                int Nb = min( SM, N - jb );
                                int Mb = min( SM, M - ib );
                                //
				//
				//printf("%d %d %d\n", ib, jb, kb);
				//
				for (kk = 0; kk < Kb; ++kk) 
				{
					for (jj = 0; jj < Nb; ++jj)
					{
						//PREFETCH1(&A[(ii + ib) + (kk + kb + 1)*lda]);
						PREFETCH1(&B[(kk + kb) + (jj + jb + 2)*ldb]);
						//PREFETCH1(&A[(kk + kb) + (jj + jb + 3)*ldb]);
						//double cij = 0.;
						//	
						for (ii = 0; ii < Mb; ++ii)
						{
							C[(ib + ii) + (jj + jb)*ldb] += A[(ii + ib) + (kk + kb)*lda]*B[(kk + kb) + (jj + jb)*ldb];
						}
					}
				}
			}
}
