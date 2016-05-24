// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file
//
//
#ifdef __GNUC__ 
#include <immintrin.h>
#include <x86intrin.h>
#warning "intrinsics loaded"
#endif

#include <sys/time.h>
#include "oncopy_4.h"
#include "otcopy_4.h"

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
#define M_BLOCK_SIZE 360
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 2880
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 360
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
	printf("%f %f %f %f\n", a[0], a[1], a[2], a[3]);

}

void print128(__m128d val)
{
        double a[2];
        _mm_store_pd(&a[0], val);
        printf("%f %f\n", a[0], a[1]);

}


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;

	double Ab[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Ap[M_BLOCK_SIZE*K_BLOCK_SIZE];
	double Bb[K_BLOCK_SIZE*N_BLOCK_SIZE]; 
        //double* Ab = (double*) malloc(M_BLOCK_SIZE*K_BLOCK_SIZE*sizeof(double));
        //double* Bb = (double*) malloc(K_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double));

/*
	double* buffsi = (double *) malloc((sizeof(double) * ((K_BLOCK_SIZE*M_BLOCK_SIZE) + (K_BLOCK_SIZE*N_BLOCK_SIZE)) + (8192)));
	double* Ab  = (double *) (((unsigned long long) buffsi + 64ULL) & ~64ULL);
	double* Bb  = (double *) ((unsigned long long) Ab + ((unsigned long long) ((((K_BLOCK_SIZE*M_BLOCK_SIZE)) * sizeof(double)) + 0xfffUL) & ~0xfffUL) + 832UL);
*/

	double copytime    = 0.;
        double computetime = 0.;
	//
	long int ops = 0;
	long int mem = 0;
	//
	register __m256d y00, y01, y02, y03;
	register __m256d y04, y05, y06, y07;
	register __m256d y08, y09, y10, y11;
	register __m256d y12, y13, y14, y15;
	//
	__m128d x00, x01, x02, x03;
	__m128d x04, x05, x06, x07;
	__m128d x08, x09, x10, x11;
	__m128d x12, x13, x14, x15;
	//
	//
	//
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
		for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
			//printf("-------> ib = %d, jb = %d, kb = %d\n", ib, jb, kb);
			copytime -= myseconds();
			oncopy_4(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
			copytime += myseconds();
			mem += Kb*Nb*8;

			for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
				copytime -= myseconds();
				otcopy_4(Kb, Mb, A + kb*lda + ib, lda, Ab);
				copytime += myseconds();
				mem += Kb*Mb*8;
				//
				int ii, kk;
				//printf("-> %d %d, %d %d\n", K_BLOCK_SIZE, M_BLOCK_SIZE, Kb, Mb); fflush(stdout);
				//double  Ap[K_BLOCK_SIZE*M_BLOCK_SIZE];
				double* pA = &Ab; 
				double* pB = &Ap;
				//
				//
				copytime -= myseconds();
				for (ii = 0; ii < Mb; ii = ii + 12)
				{
					//printf("%d\n", ii);
					pA = Ab + ii*Kb; 
					//printf("%f %f\n", *pA, *(pA + 1));
					for (kk = 0; kk < Kb; kk++)
					{
						//PREFETCH0(pA + 0*Kb +512);
						//PREFETCH0(pA + 4*Kb +512);
						//PREFETCH0(pA + 8*Kb +512);
						y00 = _mm256_load_pd(pA + 0*Kb);
						y01 = _mm256_load_pd(pA + 4*Kb);
						y02 = _mm256_load_pd(pA + 8*Kb);
						//
						//printf("addb = %d\n", addb); fflush(stdout);
						//
						_mm256_store_pd(pB + 0, y00);
						_mm256_store_pd(pB + 4, y01);
						_mm256_store_pd(pB + 8, y02);
						//
						pA   += 4;
						pB   += 12;
					}
				}
				copytime += myseconds();
				//
				computetime -= myseconds();
				//double* pA = &Ab[0];
				//double* pB = &Bb[0];
				double* pC = &C [0];
				//
				for (i = 0; i < Mb - Mb%12; i = i + 12){
					for (j = 0; j < Nb - Nb%4; j = j + 4){
						double* pB = Bb + j*Kb;
						PREFETCH2((void*) pB + 0);
						PREFETCH2((void*) pB + 8);
						//
						PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 1)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 2)*ldc + i + ib + 4]);
						PREFETCH0((void*) &C[(j + jb + 3)*ldc + i + ib + 4]);
						//
						//printf("i = %d, %d j = %d, %d\n", i, Mb, j, Nb);	
						//
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
                                                y07 = _mm256_setzero_pd();
                                                y06 = _mm256_setzero_pd();
                                                y05 = _mm256_setzero_pd();
                                                y04 = _mm256_setzero_pd();
						//
						double* pA = Ap + i*Kb + 0;
						//
						y00 = _mm256_load_pd(pA + 0);
						y01 = _mm256_load_pd(pA + 4);
						y02 = _mm256_load_pd(pA + 8);
						y03 = _mm256_load_pd(pB + 0); 
						mem += 3*8*4;
						//
						register int k = Kb >> 0;
						//
						while(k)
						{
							//printf("i = %d j = %d k = %d\n", i, j, k);
							//
							// unroll 1
							//
							PREFETCH0((void*) pB + 512);
							PREFETCH0((void*) pA + 512);
							//
							y04 = _mm256_fmadd_pd(y00, y03, y04);
							y08 = _mm256_fmadd_pd(y01, y03, y08); 
							y12 = _mm256_fmadd_pd(y02, y03, y12); 
							//
							y03 =  _mm256_permute4x64_pd(y03, 0xb1);
							y05 = _mm256_fmadd_pd(y00, y03, y05);
							y09 = _mm256_fmadd_pd(y01, y03, y09);
							y13 = _mm256_fmadd_pd(y02, y03, y13);
							//
							y03 =  _mm256_permute4x64_pd(y03, 0x1b);
							y06 = _mm256_fmadd_pd(y00, y03, y06);
							y10 = _mm256_fmadd_pd(y01, y03, y10);
							y14 = _mm256_fmadd_pd(y02, y03, y14);
							//
							y03 =  _mm256_permute4x64_pd(y03, 0xb1);
                                                        y07 = _mm256_fmadd_pd(y00, y03, y07);
							y00 = _mm256_load_pd(pA + 12);
                                                        y11 = _mm256_fmadd_pd(y01, y03, y11);	
							y01 = _mm256_load_pd(pA + 16);
                                                        y15 = _mm256_fmadd_pd(y02, y03, y15);	
							//
							//y01 = _mm256_load_pd(pA + 12);
							y02 = _mm256_load_pd(pA + 20);
							y03 = _mm256_load_pd(pB + 4);
							//
							pA += 12;
							pB += 4;
							k--;
							ops += 12*2*4;
						}
						//
						y00 = _mm256_blend_pd(y04, y05, 0xa);	
						y01 = _mm256_blend_pd(y04, y05, 0x5);	
						y02 = _mm256_blend_pd(y06, y07, 0xa);	
						y03 = _mm256_blend_pd(y06, y07, 0x5);	
						//
						y04 = _mm256_permute2f128_pd(y00, y02, 0x30);	
						y05 = _mm256_permute2f128_pd(y01, y03, 0x30);	
						y06 = _mm256_permute2f128_pd(y00, y02, 0x12);	
						y07 = _mm256_permute2f128_pd(y01, y03, 0x12);	
						//
						y00 = _mm256_blend_pd(y08, y09, 0xa);
                                                y01 = _mm256_blend_pd(y08, y09, 0x5);
                                                y02 = _mm256_blend_pd(y10, y11, 0xa);
                                                y03 = _mm256_blend_pd(y10, y11, 0x5);
                                                //
                                                y08 = _mm256_permute2f128_pd(y00, y02, 0x30);
                                                y09 = _mm256_permute2f128_pd(y01, y03, 0x30);
                                                y10 = _mm256_permute2f128_pd(y00, y02, 0x12);
                                                y11 = _mm256_permute2f128_pd(y01, y03, 0x12);
                                                //			
                                                y00 = _mm256_blend_pd(y12, y13, 0xa);
                                                y01 = _mm256_blend_pd(y12, y13, 0x5);
                                                y02 = _mm256_blend_pd(y14, y15, 0xa);
                                                y03 = _mm256_blend_pd(y14, y15, 0x5);
                                                //
                                                y12 = _mm256_permute2f128_pd(y00, y02, 0x30);
                                                y13 = _mm256_permute2f128_pd(y01, y03, 0x30);
                                                y14 = _mm256_permute2f128_pd(y00, y02, 0x12);
                                                y15 = _mm256_permute2f128_pd(y01, y03, 0x12);
                                                //
						//
						//
						y00 = _mm256_load_pd(&C[(j + jb + 0)*ldc + i + ib + 0]);
						y01 = _mm256_load_pd(&C[(j + jb + 0)*ldc + i + ib + 4]);
						y02 = _mm256_load_pd(&C[(j + jb + 0)*ldc + i + ib + 8]);
						//
						y04 = _mm256_add_pd(y00, y04);
						y08 = _mm256_add_pd(y01, y08);
						y12 = _mm256_add_pd(y02, y12);
						//
						_mm256_store_pd(&C[(j + jb + 0)*ldc + i + ib + 0], y04);
						_mm256_store_pd(&C[(j + jb + 0)*ldc + i + ib + 4], y08);
						_mm256_store_pd(&C[(j + jb + 0)*ldc + i + ib + 8], y12);
						//
						//
						//
                                                y00 = _mm256_load_pd(&C[(j + jb + 1)*ldc + i + ib + 0]);
                                                y01 = _mm256_load_pd(&C[(j + jb + 1)*ldc + i + ib + 4]);
						y02 = _mm256_load_pd(&C[(j + jb + 1)*ldc + i + ib + 8]);
                                                //
                                                y05 = _mm256_add_pd(y00, y05);
                                                y09 = _mm256_add_pd(y01, y09);
                                                y13 = _mm256_add_pd(y02, y13);
						//
						_mm256_store_pd(&C[(j + jb + 1)*ldc + i + ib + 0], y05);
                                                _mm256_store_pd(&C[(j + jb + 1)*ldc + i + ib + 4], y09);
                                                _mm256_store_pd(&C[(j + jb + 1)*ldc + i + ib + 8], y13);
						//
						//
						//
						y00 = _mm256_load_pd(&C[(j + jb + 2)*ldc + i + ib + 0]);
                                                y01 = _mm256_load_pd(&C[(j + jb + 2)*ldc + i + ib + 4]);
						y02 = _mm256_load_pd(&C[(j + jb + 2)*ldc + i + ib + 8]);
                                                //
                                                y06 = _mm256_add_pd(y00, y06);
                                                y10 = _mm256_add_pd(y01, y10);
                                                y14 = _mm256_add_pd(y02, y14);
                                                //
                                                _mm256_store_pd(&C[(j + jb + 2)*ldc + i + ib + 0], y06);
                                                _mm256_store_pd(&C[(j + jb + 2)*ldc + i + ib + 4], y10);
                                                _mm256_store_pd(&C[(j + jb + 2)*ldc + i + ib + 8], y14);
                                                //
						//
						//
                                                y00 = _mm256_load_pd(&C[(j + jb + 3)*ldc + i + ib + 0]);
                                                y01 = _mm256_load_pd(&C[(j + jb + 3)*ldc + i + ib + 4]);
						y02 = _mm256_load_pd(&C[(j + jb + 3)*ldc + i + ib + 8]);
                                                //
                                                y07 = _mm256_add_pd(y00, y07);
                                                y11 = _mm256_add_pd(y01, y11);
                                                y15 = _mm256_add_pd(y02, y15);
                                                //
                                                _mm256_store_pd(&C[(j + jb + 3)*ldc + i + ib + 0], y07);
                                                _mm256_store_pd(&C[(j + jb + 3)*ldc + i + ib + 4], y11);
                                                _mm256_store_pd(&C[(j + jb + 3)*ldc + i + ib + 8], y15);
                                                //	
						mem += 4*8*4;
						//ops += 8;
					}
				}
				computetime += myseconds();
			}
		} //
	}
//	free(buffsi);
	//free(Ab);
	//free(Bb);
	printf("%ld ops, %ld mem, copy time = %f, compute time = %f, %f GFlops\n", ops, mem, copytime, computetime, 2.*M*N*K/computetime/1e9);
}
