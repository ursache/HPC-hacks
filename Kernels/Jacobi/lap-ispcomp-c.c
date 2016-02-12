#include <omp.h>
#include <immintrin.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

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
void lap_avx(double* v1, double* v2, int dim_m, int dim_n, int lda)
{
        const int m = dim_m;
	//printf("dim_m = %d, dim_n = %d\n", dim_m, dim_n);
        //
        for (int j = 0; j < dim_n - 0; ++j )
        {
                for (int i = 0; i < dim_m - 0 - (dim_m - 0)%4; i = i + 4)
                {
                        kernel(v1 + i + j*lda, v2 + i + j*lda, lda);
                }
        }
}



void laplacian( const double * v1, 
		double * v2,
		const int dim_m, 
		const int dim_n)
{
	int offset = dim_m + 1;
	//printf("dim_n: %d\n", dim_n);
	int my_rank     = 0;
	int num_threads = 1;
#pragma omp parallel private(my_rank, num_threads) 
	{
#ifdef _OPENMP
		int my_rank     = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
#endif
		//
		//const int blocksize = (dim_n - 2)/num_threads;
		
		int n_block  = num_threads;
		int n_block_size = (int) ceil((float)(dim_n - 2)/n_block);
		int n_offset     = my_rank*n_block_size;
		n_block_size = MIN(dim_n - 2 - my_rank*n_block_size, n_block_size);
		//printf("%d: n_block = %d, n_block_size = %d, offset = %d\n", my_rank, n_block, n_block_size, offset + n_offset*dim_m);
		
		//printf("%d: ? %d %d\n", my_rank, (my_rank + 1)*n_block_size, dim_n - 2);
		//printf("blocksize = %d, offset = %d, local m = %d, where = %d\n", n_block_size, offset, dim_m - 2, offset + n_offset*dim_m);
		//const int offset_n = min(blocksize*my_rank, 
		//const int offset_m = blocksize*dim_m;
		laplacian_v6(v1 + offset + n_offset*dim_m, v2 + offset + n_offset*dim_m, dim_m - 2, n_block_size, dim_m);
		//lap_avx(v1 + offset + n_offset*dim_m, v2 + offset + n_offset*dim_m, dim_m - 2, n_block_size, dim_m);
		asm volatile ("mfence" ::: "memory");
	}
}
