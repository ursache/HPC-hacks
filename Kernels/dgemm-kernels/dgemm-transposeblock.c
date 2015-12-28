//author gilles.fourestey@epfl.ch
#define BLOCK_SIZE 64

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 256
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 3000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 256
#endif
#else
#define N_BLOCK_SIZE BLOCK_SIZE
#define M_BLOCK_SIZE BLOCK_SIZE
#define K_BLOCK_SIZE BLOCK_SIZE
#endif

#define min(a,b) (((a)<(b))?(a):(b))


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;
	int ii, jj, kk;
	//
	double a, b ,c;
	//
        double At[K*M];
        for (k = 0; k < K; ++k)
                for (i = 0; i < M; ++i)
                        At[i*M + k] = A[k*lda + i];
	//
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){
		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){
				int Kb = min( K_BLOCK_SIZE, K - kb );
				int Nb = min( N_BLOCK_SIZE, N - jb );
				int Mb = min( M_BLOCK_SIZE, M - ib );
				for (i = 0; i < Mb; i = i + 1)
					for (j = 0; j < Nb; j = j + 1)
					{
						double cij = 0.; 
						for (k = 0; k < K; ++k)
							cij += At[i*M + k]*B[k + j*ldb];
						C[j*ldc + i] += cij;
					}
			}
		}
	}
}
