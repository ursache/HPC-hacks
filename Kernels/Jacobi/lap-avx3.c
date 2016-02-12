#include <immintrin.h>

#define M_BLOCK_SIZE 512
#define N_BLOCK_SIZE 512

#define min(a,b) (((a)<(b))?(a):(b))

typedef double adouble __attribute__ ((aligned(16)));

inline
void kernel(adouble* v1, adouble * v2, int m)
{
	__m256d alpha = _mm256_set1_pd(0.25);
	//
	__m256d phi_e = _mm256_loadu_pd (v1 + 1 );
	__m256d phi_w = _mm256_loadu_pd (v1 - 1 );
	__m256d phi_n = _mm256_loadu_pd (v1 + m);
	__m256d phi_s = _mm256_loadu_pd (v1 - m);
	//
	phi_e = _mm256_add_pd(phi_e, phi_s);
	phi_e = _mm256_add_pd(phi_e, phi_n);
	//phi_e = _mm_fmadd_pd(alpha, phi_e, phi_w);
	phi_e = _mm256_add_pd(phi_e, phi_w);
	phi_e = _mm256_mul_pd(alpha, phi_e);
	//
	_mm256_storeu_pd(v2, phi_e);
}


void laplacian(double* v1, double* v2, int dim_m, int dim_n)
{
	//
	//#pragma omp parallel 
	int ib, jb;
#pragma omp parallel for schedule(static)
	for( ib = 0; ib < dim_m; ib += M_BLOCK_SIZE ){
		for( jb = 0; jb < dim_n; jb += N_BLOCK_SIZE ){
				//
				int Nb = min( N_BLOCK_SIZE, dim_n - jb );
				int Mb = min( M_BLOCK_SIZE, dim_m - ib );
				//
				for (int j = 1; j < Nb - 1; ++j)
				{
					__builtin_prefetch ((void *) v2 + j*dim_n + 256, 0, 1);
					for (int i = 1; i < Mb - 1 - (Mb - 1)%4; i = i + 4)
					{
						kernel(v1 + j*dim_n + i, v2 + j*dim_n + i, dim_n);
					}
				}
			}
}
}
