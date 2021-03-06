#include <immintrin.h>

#ifdef  __INTEL_COMPILER
inline __m128  operator+(__m128  a, __m128  b) { return _mm_add_ps(a, b); }
inline __m128i operator+(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }
inline __m128  operator*(__m128  a, __m128  b) { return _mm_mul_ps(a, b); }
inline __m128i operator*(__m128i a, __m128i b) { return _mm_mul_epu32(a, b); }
inline __m128  operator-(__m128  a, __m128  b) { return _mm_sub_ps(a, b); }
inline __m128  operator&(__m128  a, __m128  b) { return _mm_and_ps(a, b); }
inline __m128  operator|(__m128  a, __m128  b) { return _mm_or_ps(a, b); }
#endif


//typedef double adouble __attribute__ ((aligned(16)));

void kernel(double* v1, double * v2, int m)
{
	__m128d alpha = _mm_set1_pd(0.25);
	__m128d phi0, phi1;
	//
	//phi0 = phi0 + alpha*phi1;
	//
	phi0 = _mm_loadu_pd (v1 + 1 );
	phi0 = _mm_add_pd (_mm_loadu_pd(v1 - 1   ), phi0);
	phi0 = _mm_add_pd (_mm_loadu_pd(v1 + m   ), phi0);
	phi0 = _mm_add_pd (_mm_loadu_pd(v1 - m   ), phi0);
	phi0 = _mm_mul_pd(alpha, phi0);
	//
	_mm_storeu_pd(v2, phi0);
	//
}
//
void laplacian(double* v1, double* v2, int dim_m, int dim_n)
{
	int m      = dim_m;
	int offset = dim_m + 1; 
	//
	for (int j = 0; j < dim_m - 2; ++j )
	{
		for (int i = 0; i < dim_n - 2 - (dim_n - 2)%2; i = i + 2)
		{
			kernel(v1 + offset + j*dim_m + i, v2 + offset + j*dim_m + i, dim_m);
		}
	}
}

