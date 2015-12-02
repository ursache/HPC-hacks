//author gilles.fourestey@epfl.ch
#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "oncopy_4.h"
#include "otcopy_8.h"

#ifdef IACA
#include "iacaMarks.h"
#define IACAS IACA_START;
#define IACAE IACA_END;
#else
#define IACAS
#define IACAE
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
#define STORE256(A, B) _mm256_store_pd(A, B)


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
	int i, j, k;

	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
	double AB[M_BLOCK_SIZE*N_BLOCK_SIZE];

	double copytime    = 0.;
        double computetime = 0.;

	__m256d t00, t01, t02, t03;
	__m256d t04, t05, t06, t07;
	__m256d t08, t09, t10, t11;
	__m256d t12, t13, t14, t15;
	//
	__m256d y00, y01, y02, y03;
	__m256d y04, y05, y06, y07;
	__m256d y08, y09, y10, y11;
	__m256d y12, y13, y14, y15;
	//
	__m128d x00, x01, x02, x03;
	__m128d x04, x05, x06, x07;
	__m128d x08, x09, x10, x11;
	__m128d x12, x13, x14, x15;
	//
		//
		//
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
			tcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
			//
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
				//printf("-------> ib = %d, jb = %d, kb = %d\n", ib, jb, kb);
				oncopy_4(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
				//
				computetime -= myseconds();
				double* pA = &Ab[0];
				double* pB = &Bb[0];

				for (i = 0; i < Mb - Mb%8; i = i + 8){
					for (j = 0; j < Nb - Nb%4; j = j + 4){
						//
                                                double* pB = &Bb[j*Kb + 0];
                                                PREFETCH2((void*) pB + 0);
                                                PREFETCH2((void*) pB + 8);
                                                //
						PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 1)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 2)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 3)*ldc + i + ib + 4]);
                                                //

						//printf("i = %d, %d j = %d, %d\n", i, Mb, j, Nb);	
						y15 = _mm256_setzero_pd();
                                                y13 = _mm256_setzero_pd();
                                                y11 = _mm256_setzero_pd();
                                                y09 = _mm256_setzero_pd();
                                                //
						y14 = _mm256_setzero_pd();
						y12 = _mm256_setzero_pd();
						y10 = _mm256_setzero_pd();
						y08 = _mm256_setzero_pd();
						//
						double* pA = &Ab[i*Kb + 0];
						//
						y00 = _mm256_load_pd(pA + 0);
						//double* pB = &Bb[j*Kb + 0];
						//
						for (k = 0; k < Kb - Kb%1; k = k + 1)
						{
							IACAS;
							//printf("i = %d j = %d k = %d\n", i, j, k);
							PREFETCH0((void*) pB + 256);
							y02 = _mm256_broadcast_sd(pB + 0);	// y02 = B[0]
							y01 = _mm256_load_pd(pA + 4); 		// y00 = A[0]
							//
							y06 = _mm256_mul_pd(y00, y02);		// y06 = A[0]*B[0]
							y03 = _mm256_broadcast_sd(pB + 1);	// y03 = B[1] 
							y04 = _mm256_broadcast_sd(pB + 2);	// y04 = B[2] 
							y07 = _mm256_mul_pd(y00, y03);		// y07 = A[0]*B[1]
							//
							y15 = _mm256_add_pd(y15, y06);		// y15 = y15 + A[0]*B[0]
							y13 = _mm256_add_pd(y13, y07); 		// y13 = y13 + A[0]*B[1]
							//
							y06 = _mm256_mul_pd(y00, y04);		// y06 = A[0]*B[2] 
							//	
							y05 = _mm256_broadcast_sd(pB + 3);	// y05 = B[3] 
							y07 = _mm256_mul_pd(y00, y05);		// y07 = A[0]*B[3]
							y00 = _mm256_load_pd(pA + 8);		// y01 = A[4]
							//
							y11 = _mm256_add_pd(y11, y06);		// y11 = y11 + A[0]*B[2]
							y09 = _mm256_add_pd(y09, y07);		// y09 = y09 + A[0]*B[3]
							//print256(y11);
							//
							PREFETCH0((void*) pA + 512);
							//
							//
							y06 = _mm256_mul_pd(y01, y02);		// y06 = A[4]*B[0]
							y07 = _mm256_mul_pd(y01, y03);		// y07 = A[4]*B[1]
							//
							y14 = _mm256_add_pd(y14, y06);		// y14 = y14 + A[4]*B[0]
							y12 = _mm256_add_pd(y12, y07);		// y12 = y12 + A[4]*B[1]
							//
							y06 = _mm256_mul_pd(y01, y04);		// y06 = A[4]*B[2]
							y07 = _mm256_mul_pd(y01, y05);		// y07 = A[4]*B[3]
							//
							y10 = _mm256_add_pd(y10, y06);		// y10 = y10 + A[4]*B[2]
							y08 = _mm256_add_pd(y08, y07);		// y08 = y08 + A[4]*B[3]
							//
							pA += 8;
							pB += 4;
							//
							IACAE;
						}
						//
						y07 = _mm256_load_pd(&C[(j + jb + 0)*ldc + i + ib + 0]);
						y05 = _mm256_load_pd(&C[(j + jb + 1)*ldc + i + ib + 0]);
						y03 = _mm256_load_pd(&C[(j + jb + 2)*ldc + i + ib + 0]);
						y01 = _mm256_load_pd(&C[(j + jb + 3)*ldc + i + ib + 0]);
						//
						y07 = _mm256_add_pd(y07, y15);
						y05 = _mm256_add_pd(y05, y13);
						y03 = _mm256_add_pd(y03, y11);
						y01 = _mm256_add_pd(y01, y09);
						//
						STORE256(&C[(j + jb + 0)*ldc + i + ib + 0], y07);
                                                STORE256(&C[(j + jb + 1)*ldc + i + ib + 0], y05);
                                                STORE256(&C[(j + jb + 2)*ldc + i + ib + 0], y03);
                                                STORE256(&C[(j + jb + 3)*ldc + i + ib + 0], y01);
						//
						y06 = _mm256_load_pd(&C[(j + jb + 0)*ldc + i + ib + 4]);
                                                y04 = _mm256_load_pd(&C[(j + jb + 1)*ldc + i + ib + 4]);
                                                y02 = _mm256_load_pd(&C[(j + jb + 2)*ldc + i + ib + 4]);
                                                y00 = _mm256_load_pd(&C[(j + jb + 3)*ldc + i + ib + 4]);
						//
						y06 = _mm256_add_pd(y06, y14);
                                                y04 = _mm256_add_pd(y04, y12);
                                                y02 = _mm256_add_pd(y02, y10);
                                                y00 = _mm256_add_pd(y00, y08);
						//
						STORE256(&C[(j + jb + 0)*ldc + i + ib + 4], y06);
                                                STORE256(&C[(j + jb + 1)*ldc + i + ib + 4], y04);
                                                STORE256(&C[(j + jb + 2)*ldc + i + ib + 4], y02);
                                                STORE256(&C[(j + jb + 3)*ldc + i + ib + 4], y00);
						//
					}
				}
				computetime += myseconds();
			}
		} //
	}
	printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}
