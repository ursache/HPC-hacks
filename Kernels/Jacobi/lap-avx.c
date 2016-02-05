#include <immintrin.h>

typedef double adouble __attribute__ ((aligned(16)));

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
	int m = dim_m;
	//
	//#pragma omp parallel 
#pragma omp parallel for schedule(static)
	for (int j = 1; j < dim_m - 1; ++j )
	{
		for (int i = 1; i < dim_n - 1 - (dim_n - 1)%4; i = i + 4)
		{
			kernel(v1 + j*dim_m + i, v2 + j*dim_m + i, dim_m);
		}
	}
}
