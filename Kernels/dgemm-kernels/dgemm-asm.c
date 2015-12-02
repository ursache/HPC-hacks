//author gilles.fourestey@epfl.ch
#ifdef __GNUC__ 
#include <immintrin.h>
//#include <x86intrin.h>
#endif

#include <sys/time.h>
#include "oncopy.h"
#include "otcopy.h"
#include "otcopy_8.h"

#ifdef IACA
#include "iacaMarks.h"
#endif

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 256
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 32
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 256
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

	double* Ab = _mm_malloc(M_BLOCK_SIZE*K_BLOCK_SIZE*sizeof(double), 32);
	double* Bb = _mm_malloc(K_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double), 32); 
	double* AB = _mm_malloc(M_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double), 32);
	
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
			copytime -= myseconds();
			otcopy_8(Kb, Mb, A + kb*lda + ib, lda, Ab);
			copytime += myseconds();
#if 0
			printf("\n");
			for(i = 0; i < Mb; ++i)
			{	
				for(k = 0; k < Kb; ++k)
				{
					//Ab[i*K_BLOCK_SIZE + k] = A[(k + kb)*lda + (i + ib)];
					//Ab[k*M_BLOCK_SIZE + i] = A[(k + kb)*lda + (i + ib)];
					printf("%f ", A[k*lda + i]);
				}
				printf("\n");
			}
			printf("\n");
			for(i = 0; i < Mb; ++i)
			{
				for(k = 0; k < Kb; ++k)
				{
					//Ab[i*K_BLOCK_SIZE + k] = A[(k + kb)*lda + (i + ib)];
					//                                        //Ab[k*M_BLOCK_SIZE + i] = A[(k + kb)*lda + (i + ib)];
					printf("%f\n", Ab[i*Kb + k]);
				}
				printf("\n");
			}
			printf("\n");
#endif
			//
			for( jb = 0; jb < N; jb += N_BLOCK_SIZE ){ int Nb = min( N_BLOCK_SIZE, N - jb );
				//printf("-------> ib = %d, jb = %d, kb = %d\n", ib, jb, kb);
				copytime -= myseconds();
				oncopy(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
				copytime += myseconds();
				
#if 0
				for (k = 0; k < Kb - Kb%1; k = k + 1){
					for (j = 0; j < Nb; j = j + 1){
						printf("%f ", B[j*ldb + k]);
					}
					printf("\n");
				}
				printf("\n");
				for (j = 0; j < Nb; j = j + 1){
					for (k = 0; k < Kb - Kb%1; k = k + 1){
						printf("%f\n", Bb[j*Kb + k]);
					}
					printf("\n");
				}
#endif

				computetime -= myseconds();
				//double* pA = &Ab[0];
				//double* pB = &Bb[0];
				double* pC = &C[0];
				//
				//printf("C = %p\n", C);
				for (i = 0; i < Mb - Mb%8; i = i + 8){
					for (j = 0; j < Nb - Nb%4; j = j + 4)
					{
						//printf("i = %d, %d j = %d, %d ldc = %d\n", i, ib, j, jb, ldc);	
						printf("i*Kb = %d\n", i*Kb);
						double* pA = Ab + i*Kb + 0;
						double* pB = Bb + j*Kb + 0;
						pC = C  + (j + jb)*ldc + i + ib;
						//
						PREFETCH2((void*) pB + 0);
						PREFETCH2((void*) pB + 8);
						//
						PREFETCH0((void*) &C[(j + jb + 0)*ldc + i + ib + 0]);
						PREFETCH0((void*) &C[(j + jb + 1)*ldc + i + ib + 0]);
						PREFETCH0((void*) &C[(j + jb + 2)*ldc + i + ib + 0]);
						PREFETCH0((void*) &C[(j + jb + 3)*ldc + i + ib + 0]);
						//
						//y15 = _mm256_setzero_pd(); 
						//y13 = _mm256_setzero_pd(); 
						//y11 = _mm256_setzero_pd(); 
						//y09 = _mm256_setzero_pd(); 
						//
						//y14 = _mm256_setzero_pd(); 
						//y12 = _mm256_setzero_pd(); 
						//y10 = _mm256_setzero_pd(); 
						//y08 = _mm256_setzero_pd(); 
						//
						//printf("pA = %p\n", pA);
						//printf("pB = %p\n", pB);
						//printf("pC = %p\n", pC);
						//
						//y00 = _mm256_load_pd(pA + 0);
						//y02 = _mm256_load_pd(pB + 0);
						//y03 = _mm256_permute_pd(y02, 5);
						//
						//k = Kb;
						//printf("k = %d, k >> 2 = %d\n", k, k >> 2); 

						//
						//for (k = 0; k < Kb - Kb%1 - 1; k = k + 1)
						//						while(k)
						//						
						k = Kb >> 0;
						//printf("i = %d j = %d k = %d\n", i, j, k);
						//
						// unroll 1
						//
#if 1
						__asm__ __volatile__(
								"movq %4,%%r10;"
								"lea (, %%r10, 0x8), %%r10;"
								"movq %3, %%r11;"
								"lea (%%r11, %%r10, 0x2),%%r12;"  
								".align (32);"
								"vxorpd %%ymm15,%%ymm15,%%ymm15;"
								"vxorpd %%ymm14,%%ymm14,%%ymm14;"
								"vxorpd %%ymm13,%%ymm13,%%ymm13;"
								"vxorpd %%ymm12,%%ymm12,%%ymm12;"
								"vxorpd %%ymm11,%%ymm11,%%ymm11;"
								"vxorpd %%ymm10,%%ymm10,%%ymm10;"
								"vxorpd %%ymm9,%%ymm9,%%ymm9;"
								"vxorpd %%ymm8,%%ymm8,%%ymm8;"
								"vmovupd (%%rsi),%%ymm2;"
								"vmovupd (%%rdi),%%ymm0;"
								"vpermilpd $0x5,%%ymm2,%%ymm3;"
								".align(32);"
								"k_loop:;"
								//
								"vmovupd 0x20(%%rdi),%%ymm1;"
								"vmulpd %%ymm0,%%ymm2,%%ymm6;"
								"vperm2f128 $0x3,%%ymm2,%%ymm2,%%ymm4;"
								"vmulpd %%ymm0,%%ymm3,%%ymm7;"
								"vperm2f128 $0x3,%%ymm3,%%ymm3,%%ymm5;"
								"vaddpd %%ymm15,%%ymm6,%%ymm15;"
								"vaddpd %%ymm13,%%ymm7,%%ymm13;"
								"prefetcht0 0x200(%%rdi);"
								"vmulpd %%ymm1,%%ymm2,%%ymm6;"
								"vmovupd 0x20(%%rsi),%%ymm2;"
								"vmulpd %%ymm1,%%ymm3,%%ymm7;"
								"vpermilpd $0x5,%%ymm2,%%ymm3;"
								"vaddpd %%ymm14,%%ymm6,%%ymm14;"
								"vaddpd %%ymm12,%%ymm7,%%ymm12;"
								"vmulpd %%ymm0,%%ymm4,%%ymm6;"
								"vmulpd %%ymm0,%%ymm5,%%ymm7;"
								"vmovupd 0x40(%%rdi),%%ymm0;"
								"vaddpd %%ymm11,%%ymm6,%%ymm11;"
								"vaddpd %%ymm9,%%ymm7,%%ymm9;"
								"vmulpd %%ymm1,%%ymm4,%%ymm6;"
								"vmulpd %%ymm1,%%ymm5,%%ymm7;"
								"vaddpd %%ymm10,%%ymm6,%%ymm10;"
								"vaddpd %%ymm8,%%ymm7,%%ymm8;"
								"add $0x40, %%rdi;"
                                                                "add $0x20, %%rsi;"
								/*
								"vmovupd 0x60(%%rdi),%%ymm1;"
                                                                "vmulpd %%ymm0,%%ymm2,%%ymm6;"
                                                                "vperm2f128 $0x3,%%ymm2,%%ymm2,%%ymm4;"
                                                                "vmulpd %%ymm0,%%ymm3,%%ymm7;"
                                                                "vperm2f128 $0x3,%%ymm3,%%ymm3,%%ymm5;"
                                                                "vaddpd %%ymm15,%%ymm6,%%ymm15;"
                                                                "vaddpd %%ymm13,%%ymm7,%%ymm13;"
                                                                "prefetcht0 0x240(%%rdi);"
                                                                "vmulpd %%ymm1,%%ymm2,%%ymm6;"
                                                                "vmovupd 0x60(%%rsi),%%ymm2;"
                                                                "vmulpd %%ymm1,%%ymm3,%%ymm7;"
                                                                "vpermilpd $0x5,%%ymm2,%%ymm3;"
                                                                "vaddpd %%ymm14,%%ymm6,%%ymm14;"
                                                                "vaddpd %%ymm12,%%ymm7,%%ymm12;"
                                                                "vmulpd %%ymm0,%%ymm4,%%ymm6;"
                                                                "vmulpd %%ymm0,%%ymm5,%%ymm7;"
                                                                "vmovupd 0x80(%%rdi),%%ymm0;"
                                                                "vaddpd %%ymm11,%%ymm6,%%ymm11;"
                                                                "vaddpd %%ymm9,%%ymm7,%%ymm9;"
                                                                "vmulpd %%ymm1,%%ymm4,%%ymm6;"
                                                                "vmulpd %%ymm1,%%ymm5,%%ymm7;"
                                                                "vaddpd %%ymm10,%%ymm6,%%ymm10;"
                                                                "vaddpd %%ymm8,%%ymm7,%%ymm8;"
								"add $0x40, %%rdi;"
								"add $0x20, %%rsi;" 
								*/
								"dec %%rax;"
								"jg k_loop;"
								"vmovapd %%ymm15,%%ymm7;"
								"vshufpd $0xa,%%ymm13,%%ymm15,%%ymm15;"
								"vshufpd $0xa,%%ymm7,%%ymm13,%%ymm13;"
								"vmovapd %%ymm14,%%ymm7;"
								"vshufpd $0xa,%%ymm12,%%ymm14,%%ymm14;"
								"vshufpd $0xa,%%ymm7,%%ymm12,%%ymm12;"
								"vmovapd %%ymm11,%%ymm7;"
								"vshufpd $0xa,%%ymm9,%%ymm11,%%ymm11;"
								"vshufpd $0xa,%%ymm7,%%ymm9,%%ymm9;"
								"vmovapd %%ymm10,%%ymm7;"
								"vshufpd $0xa,%%ymm8,%%ymm10,%%ymm10;"
								"vshufpd $0xa,%%ymm7,%%ymm8,%%ymm8;"
								".align (32);"
								"vextractf128 $0x1,%%ymm15,%%xmm7;"
								"vextractf128 $0x1,%%ymm14,%%xmm6;"
								"vextractf128 $0x1,%%ymm13,%%xmm5;"
								"vextractf128 $0x1,%%ymm12,%%xmm4;"
								"vextractf128 $0x1,%%ymm11,%%xmm3;"
								"vextractf128 $0x1,%%ymm10,%%xmm2;"
								"vextractf128 $0x1,%%ymm9,%%xmm1;"
								"vextractf128 $0x1,%%ymm8,%%xmm0;"
/*
								"vaddpd (%%r11),%%xmm15,%%xmm15;"
								"vaddpd 0x10(%%r12),%%xmm7,%%xmm7;"
								"vaddpd 0x20(%%r11),%%xmm14,%%xmm14;"
								"vaddpd 0x30(%%r12),%%xmm6,%%xmm6;"
								"vaddpd (%%r11,%%r10,1),%%xmm13,%%xmm13;"
								"vaddpd 0x10(%%r12,%%r10,1),%%xmm5,%%xmm5;"
								"vaddpd 0x20(%%r11,%%r10,1),%%xmm12,%%xmm12;"
								"vaddpd 0x30(%%r12,%%r10,1),%%xmm4,%%xmm4;"
								"vaddpd 0x0(%%r12),%%xmm11,%%xmm11;"
								"vaddpd 0x10(%%r11),%%xmm3,%%xmm3;"
								"vaddpd 0x20(%%r12),%%xmm10,%%xmm10;"
								"vaddpd 0x30(%%r11),%%xmm2,%%xmm2;"
								"vaddpd 0x0(%%r12,%%r10,1),%%xmm9,%%xmm9;"
								"vaddpd 0x10(%%r11,%%r10,1),%%xmm1,%%xmm1;"
								"vaddpd 0x20(%%r12,%%r10,1),%%xmm8,%%xmm8;"
								"vaddpd 0x30(%%r11,%%r10,1),%%xmm0,%%xmm0;"
*/
								"vmovupd %%xmm15,(%%r11);"
								"vmovupd %%xmm7,0x10(%%r12);"
								"vmovupd %%xmm14,0x20(%%r11);"
								"vmovupd %%xmm6,0x30(%%r12);"
								"vmovupd %%xmm13,(%%r11,%%r10,1);"
								"vmovupd %%xmm5,0x10(%%r12,%%r10,1);"
								"vmovupd %%xmm12,0x20(%%r11,%%r10,1);"
								"vmovupd %%xmm4,0x30(%%r12,%%r10,1);"
								"vmovupd %%xmm11,0x0(%%r12);"
								"vmovupd %%xmm3,0x10(%%r11);"
								"vmovupd %%xmm10,0x20(%%r12);"
								"vmovupd %%xmm2,0x30(%%r11);"
								"vmovupd %%xmm9,0x0(%%r12,%%r10,1);"
								"vmovupd %%xmm1,0x10(%%r11,%%r10,1);"
								"vmovupd %%xmm8,0x20(%%r12,%%r10,1);"
								"vmovupd %%xmm0,0x30(%%r11,%%r10,1);"
								: 
								: "D"(pA), "S"(pB), "a"(k), "m"(pC), "m"(ldc)
								: "%r10", "%r11", "%r12" 
								//: "%rbp" 
								    );
#else
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
                                                y00 = _mm256_load_pd(pA + 0);
                                                y02 = _mm256_load_pd(pB + 0);
                                                y03 = _mm256_permute_pd(y02, 5);
                                                //
                                                //k = Kb;
                                                //printf("k = %d, k >> 2 = %d\n", k, k >> 2); 
                                                //
                                                //for (k = 0; k < Kb - Kb%1 - 1; k = k + 1)
                                                while(k)
                                                {
                                                        //printf("i = %d j = %d k = %d\n", i, j, k);
                                                        //
                                                        // unroll 1
                                                        //
                                                        PREFETCH0((void*) pB + 64);
                                                        y01 = _mm256_load_pd(pA + 4);
                                                        //
                                                        y06 = _mm256_mul_pd(y00, y02);
                                                        y04 = _mm256_permute2f128_pd(y02, y02, 0x3);
                                                        y07 = _mm256_mul_pd(y00, y03);
                                                        y05 = _mm256_permute2f128_pd(y03, y03, 0x3);
                                                        y15 = _mm256_add_pd(y15, y06);
                                                        y13 = _mm256_add_pd(y13, y07);
                                                        //
                                                        PREFETCH0((void*) pA + 64);
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
							pA += 8;
                                                        pB += 4;
							k--;
						}
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
#endif
					} 
					}
				computetime += myseconds();
				}
			} //
		}
		_mm_free(Ab);	
		_mm_free(Bb);	
		_mm_free(AB);	
		//printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
	}
