#include <immintrin.h>

inline
void kernel(double* v1, double * v2, int m)
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


inline
void kernel_sequential(double* v1, double * v2, int m)
{
        double phi_e = *(v1 + 1);
        double phi_w = *(v1 - 1);

        double phi_n = *(v1 + m);
        double phi_s = *(v1 - m);

        double phi = 0.25*(phi_e + phi_w + phi_n + phi_s);

        *(v2) = phi;
}



void laplacian(double* v1, double* v2, int dim_m, int dim_n)
{
	//
	//#pragma omp parallel 
#pragma omp parallel for schedule(static)
	for (int j = 1; j < dim_n - 1; ++j )
	{
		__builtin_prefetch ((void *) v2 + j*dim_n + 256, 0, 1);
		int i;
		for (i = 1; i < dim_m - 1 - (dim_m - 1)%4; i = i + 4)
		{
			kernel(v1 + j*dim_n + i, v2 + j*dim_n + i, dim_n);
		}
		for (; i < dim_m - 1; ++i)
		{
			kernel_sequential(v1 + j*dim_n + i, v2 + j*dim_n + i, dim_n);
		}
	}
}
