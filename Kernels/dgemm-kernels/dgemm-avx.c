// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file
//

#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "oncopy_4.h"
//#include "otcopy.h"
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
#define M_BLOCK_SIZE 240
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 3000
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 240
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
	//
	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
	//
	double copytime    = 0.;
        double computetime = 0.;
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
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
		for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
			//printf("-------> ib = %d, jb = %d, kb = %d\n", ib, jb, kb);
			oncopy_4(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
			//
			for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
				//
				tcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
				//
				computetime -= myseconds();
				//double* pA = &Ab[0];
				//double* pB = &Bb[0];
				double* pC = &C [0];
				//
				for (i = 0; i < Mb - Mb%8; i = i + 8){
					for (j = 0; j < Nb - Nb%4; j = j + 4){
						//
						register double* pB = &Bb[j*Kb + 0];
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
						register double* pA = &Ab[i*Kb + 0];
						//
						y00 = _mm256_load_pd(pA + 0);
						y02 = _mm256_load_pd(pB + 0);
						y03 = _mm256_permute_pd(y02, 5);
						//
						//k = Kb;
						//printf("k = %d, k >> 2 = %d\n", k, k >> 2); 
						register int k = Kb >> 0;
						//
						//for (k = 0; k < Kb - Kb%1 - 1; k = k + 1)
						while(k)
						{
							//printf("i = %d j = %d k = %d\n", i, j, k);
							//
							// unroll 1
							//
							PREFETCH0((void*) pA + 512);
							PREFETCH0((void*) pB + 512);
							y01 = _mm256_load_pd(pA + 4);
							//
							y06 = _mm256_mul_pd(y00, y02);
							y04 = _mm256_permute2f128_pd(y02, y02, 0x3);
							y07 = _mm256_mul_pd(y00, y03);
							y05 = _mm256_permute2f128_pd(y03, y03, 0x3);
							y15 = _mm256_add_pd(y15, y06);
							y13 = _mm256_add_pd(y13, y07); 
							//
							y06 = _mm256_mul_pd(y01, y02);
							y02 = _mm256_load_pd(pB + 4);
							y07 = _mm256_mul_pd(y01, y03);
							y03 = _mm256_permute_pd(y02, 5);
							y14 = _mm256_add_pd(y14, y06);
							y12 = _mm256_add_pd(y12, y07);
							//
							y06 = _mm256_mul_pd(y00, y04);
							y07 = _mm256_mul_pd(y00, y05);
							y00 = _mm256_load_pd(pA + 8);
							y11 = _mm256_add_pd(y11, y06);
							y09 = _mm256_add_pd(y09, y07);
							//
							y06 = _mm256_mul_pd(y01, y04);
							y07 = _mm256_mul_pd(y01, y05);
							y10 = _mm256_add_pd(y10, y06);
							y08 = _mm256_add_pd(y08, y07);
							//
							// unroll 2
							/*
							   y01 = _mm256_load_pd(pA + 12);
							//
							y06 = _mm256_mul_pd(y00, y02);
							y04 = _mm256_permute2f128_pd(y02, y02, 0x3);
							y07 = _mm256_mul_pd(y00, y03);
							y05 = _mm256_permute2f128_pd(y03, y03, 0x3);
							y15 = _mm256_add_pd(y15, y06);
							y13 = _mm256_add_pd(y13, y07);
							//
							PREFETCH0((void*) pA + 72);
							y06 = _mm256_mul_pd(y01, y02);
							y02 = _mm256_load_pd(pB + 8);
							y07 = _mm256_mul_pd(y01, y03);
							y03 = _mm256_permute_pd(y02, 5);
							y14 = _mm256_add_pd(y14, y06);
							y12 = _mm256_add_pd(y12, y07);
							//
							y06 = _mm256_mul_pd(y00, y04);
							y07 = _mm256_mul_pd(y00, y05);
							y00 = _mm256_load_pd(pA + 16);
							y11 = _mm256_add_pd(y11, y06);
							y09 = _mm256_add_pd(y09, y07);
							//
							y06 = _mm256_mul_pd(y01, y04);
							y07 = _mm256_mul_pd(y01, y05);
							y10 = _mm256_add_pd(y10, y06);
							y08 = _mm256_add_pd(y08, y07);
							//
							// unroll 3
							//
							y01 = _mm256_load_pd(pA + 20);
							//
							y06 = _mm256_mul_pd(y00, y02);
							y04 = _mm256_permute2f128_pd(y02, y02, 0x3);
							y07 = _mm256_mul_pd(y00, y03);
							y05 = _mm256_permute2f128_pd(y03, y03, 0x3);
							y15 = _mm256_add_pd(y15, y06);
							y13 = _mm256_add_pd(y13, y07);
							//
							PREFETCH0((void*) pA + 80);
							y06 = _mm256_mul_pd(y01, y02);
							y02 = _mm256_load_pd(pB + 12);
							pB += 16;
							y07 = _mm256_mul_pd(y01, y03);
							y03 = _mm256_permute_pd(y02, 5);
							y14 = _mm256_add_pd(y14, y06);
							y12 = _mm256_add_pd(y12, y07);
							//
							y06 = _mm256_mul_pd(y00, y04);
							y07 = _mm256_mul_pd(y00, y05);
							y00 = _mm256_load_pd(pA + 24);
							y11 = _mm256_add_pd(y11, y06);
							y09 = _mm256_add_pd(y09, y07);
							//
							y06 = _mm256_mul_pd(y01, y04);
							y07 = _mm256_mul_pd(y01, y05);
							y10 = _mm256_add_pd(y10, y06);
							y08 = _mm256_add_pd(y08, y07);
							//
							// unroll 4
							//
							y01 = _mm256_load_pd(pA + 28);
							//
							y06 = _mm256_mul_pd(y00, y02);
							y04 = _mm256_permute2f128_pd(y02, y02, 0x3);
							y07 = _mm256_mul_pd(y00, y03);
							y05 = _mm256_permute2f128_pd(y03, y03, 0x3);
							pA += 32;
							y15 = _mm256_add_pd(y15, y06);
							y13 = _mm256_add_pd(y13, y07);
							//
							PREFETCH0((void*) pA + 88);
							y06 = _mm256_mul_pd(y01, y02);
							y02 = _mm256_load_pd(pB + 0);
							y07 = _mm256_mul_pd(y01, y03);
							y03 = _mm256_permute_pd(y02, 5);
							y14 = _mm256_add_pd(y14, y06);
							y12 = _mm256_add_pd(y12, y07);
							//
							y06 = _mm256_mul_pd(y00, y04);
							y07 = _mm256_mul_pd(y00, y05);
							y00 = _mm256_load_pd(pA + 8);
							y11 = _mm256_add_pd(y11, y06);
							y09 = _mm256_add_pd(y09, y07);
							//
							y06 = _mm256_mul_pd(y01, y04);
							y07 = _mm256_mul_pd(y01, y05);
							y10 = _mm256_add_pd(y10, y06);
							y08 = _mm256_add_pd(y08, y07);
							*/	
								pA += 8;
							pB += 4;
							k--;
						}
						//
						//
						y07 = y15;
						y15 = _mm256_shuffle_pd(y15, y13, 0xa);
						y13 = _mm256_shuffle_pd(y13, y07, 0xa);
						//
						y07 = y11;
						y11 = _mm256_shuffle_pd(y11, y09, 0xa);
						y09 = _mm256_shuffle_pd(y09, y07, 0xa);
						//
						x07 = _mm_load_pd(&C[(j + jb + 2)*ldc + i + ib + 2]);
						x05 = _mm_load_pd(&C[(j + jb + 3)*ldc + i + ib + 2]);
						x03 = _mm_load_pd(&C[(j + jb + 0)*ldc + i + ib + 2]);
						x01 = _mm_load_pd(&C[(j + jb + 1)*ldc + i + ib + 2]);
						//
						x07 = _mm_add_pd(_mm256_extractf128_pd(y15, 0x1), x07);
						x05 = _mm_add_pd(_mm256_extractf128_pd(y13, 0x1), x05);
						x03 = _mm_add_pd(_mm256_extractf128_pd(y11, 0x1), x03);
						x01 = _mm_add_pd(_mm256_extractf128_pd(y09, 0x1), x01);
						//
						STORE128(&C[(j + jb + 2)*ldc + i + ib + 2], x07);
						STORE128(&C[(j + jb + 3)*ldc + i + ib + 2], x05);
						STORE128(&C[(j + jb + 0)*ldc + i + ib + 2], x03);
						STORE128(&C[(j + jb + 1)*ldc + i + ib + 2], x01);
						//
						x07 = _mm_load_pd(&C[(j + jb + 0)*ldc + i + ib + 0]);
						x05 = _mm_load_pd(&C[(j + jb + 1)*ldc + i + ib + 0]);
						x03 = _mm_load_pd(&C[(j + jb + 2)*ldc + i + ib + 0]);
						x01 = _mm_load_pd(&C[(j + jb + 3)*ldc + i + ib + 0]);
						//
						x07 = _mm_add_pd(_mm256_extractf128_pd(y15, 0x0), x07);
						x05 = _mm_add_pd(_mm256_extractf128_pd(y13, 0x0), x05);
						x03 = _mm_add_pd(_mm256_extractf128_pd(y11, 0x0), x03);
						x01 = _mm_add_pd(_mm256_extractf128_pd(y09, 0x0), x01);
						//
						STORE128(&C[(j + jb + 0)*ldc + i + ib + 0], x07);
						STORE128(&C[(j + jb + 1)*ldc + i + ib + 0], x05);
						STORE128(&C[(j + jb + 2)*ldc + i + ib + 0], x03);
						STORE128(&C[(j + jb + 3)*ldc + i + ib + 0], x01);
						//
						// second part
						//
						y07 = y14;
						y14 = _mm256_shuffle_pd(y14, y12, 0xa);
						y12 = _mm256_shuffle_pd(y12, y07, 0xa);
						//
						y07 = y10;
						y10 = _mm256_shuffle_pd(y10, y08, 0xa);
						y08 = _mm256_shuffle_pd(y08, y07, 0xa);
						//
						x06 = _mm_load_pd(&C[(j + jb + 2)*ldc + i + ib + 6]);
						x04 = _mm_load_pd(&C[(j + jb + 3)*ldc + i + ib + 6]);
						x02 = _mm_load_pd(&C[(j + jb + 0)*ldc + i + ib + 6]);
						x00 = _mm_load_pd(&C[(j + jb + 1)*ldc + i + ib + 6]);
						//
						x06 = _mm_add_pd(_mm256_extractf128_pd(y14, 0x1), x06);
						x04 = _mm_add_pd(_mm256_extractf128_pd(y12, 0x1), x04);
						x02 = _mm_add_pd(_mm256_extractf128_pd(y10, 0x1), x02);
						x00 = _mm_add_pd(_mm256_extractf128_pd(y08, 0x1), x00);
						//
						STORE128(&C[(j + jb + 2)*ldc + i + ib + 6], x06);
						STORE128(&C[(j + jb + 3)*ldc + i + ib + 6], x04);
						STORE128(&C[(j + jb + 0)*ldc + i + ib + 6], x02);
						STORE128(&C[(j + jb + 1)*ldc + i + ib + 6], x00);
						//
						x06 = _mm_load_pd(&C[(j + jb + 0)*ldc + i + ib + 4]);
						x04 = _mm_load_pd(&C[(j + jb + 1)*ldc + i + ib + 4]);
						x02 = _mm_load_pd(&C[(j + jb + 2)*ldc + i + ib + 4]);
						x00 = _mm_load_pd(&C[(j + jb + 3)*ldc + i + ib + 4]);
						//
						x06 = _mm_add_pd(_mm256_extractf128_pd(y14, 0x0), x06);
						x04 = _mm_add_pd(_mm256_extractf128_pd(y12, 0x0), x04);
						x02 = _mm_add_pd(_mm256_extractf128_pd(y10, 0x0), x02);
						x00 = _mm_add_pd(_mm256_extractf128_pd(y08, 0x0), x00);
						//
						STORE128(&C[(j + jb + 0)*ldc + i + ib + 4], x06);
						STORE128(&C[(j + jb + 1)*ldc + i + ib + 4], x04);
						STORE128(&C[(j + jb + 2)*ldc + i + ib + 4], x02);
						STORE128(&C[(j + jb + 3)*ldc + i + ib + 4], x00);
						//
					}
				}
				computetime += myseconds();
			}
		} //
	}
	//	free(buffsi);
	printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}
