//author gilles.fourestey@epfl.ch
//
// author gilles.fourestey@epfl.ch 
// 
#ifdef __GNUC__ 
//#ifdef __MIC__
#include <immintrin.h>
#warning "including zmmintrin.h"
//#include <zmmintrin.h>
//#include <x86intrin.h>
//#endif
#endif

#include <sys/time.h>
//#include "oncopy.h"
#include "oncopy_16.h"
//#include "otcopy.h"
#include "otcopy_8.h"

#ifdef IACA
#include "iacaMarks.h"
#endif

#if !defined(BLOCK_SIZE)
#ifndef M_BLOCK_SIZE
#define M_BLOCK_SIZE 320
#endif
#ifndef N_BLOCK_SIZE
#define N_BLOCK_SIZE 16
#endif
#ifndef K_BLOCK_SIZE
#define K_BLOCK_SIZE 320
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


//#define STORE128(A, B) _mm_stream_pd(A, B)
#define __m512d	__m512d
#define REAL256	__m256d
#define REAL128 __m128d

#define STORE512(A, B) _mm512_store_pd(A, B)
#define STORE256(A, B) _mm256_store_pd(A, B)
#define STORE128(A, B) _mm_store_pd(A, B)

#define LOAD512(A)     _mm512_load_pd(A)
#define LOAD256(A)     _mm256_load_pd(A)
#define LOAD128(A)     _mm_load_pd(A)

#define MUL512(A, B)   _mm512_mul_pd(A, B)
#define MUL256(A, B)   _mm256_mul_pd(A, B)
#define MUL128(A, B)   _mm128_mul_pd(A, B)

#define ADD512(A, B)   _mm512_add_pd(A, B)
#define ADD256(A, B)   _mm256_add_pd(A, B)
#define ADD128(A, B)   _mm128_add_pd(A, B)

double
__attribute__((target(mic)))
myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void
__attribute__((target(mic)))
print512(__m512d val){
#ifdef __MIC
	double a[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
	STORE512(&a[0], val);
	printf("%f %f %f %f %f %f %f %f", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
#else
	printf("print512: function only defined for MIC!\n");
#endif
}

void
__attribute__((target(mic)))
dgemm_knc( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int ib, jb, kb;
	int i, j, k;

	double* Ab = _mm_malloc(M_BLOCK_SIZE*K_BLOCK_SIZE*sizeof(double), 32);
	double* Bb = _mm_malloc(K_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double), 32); 
	double* AB = _mm_malloc(M_BLOCK_SIZE*N_BLOCK_SIZE*sizeof(double), 32);

//        int kstart = 0;
//        while ( ((long) &C[kstart]) & 0x000000000000003F ) kstart++;
//        printf("C = %p, alignement = %x %x\n", C, 8*kstart, 64);

	double copytime    = 0.;
        double computetime = 0.;

	__m512d t00, t01, t02, t03;
	__m512d t04, t05, t06, t07;
	__m512d t08, t09, t10, t11;
	__m512d t12, t13, t14, t15;
	//
	__m512d y00, y01, y02, y03;
	__m512d y04, y05, y06, y07;
	__m512d y08, y09, y10, y11;
	__m512d y12, y13, y14, y15;
	__m512d y16, y17, y18, y19;
	__m512d y20, y21, y22, y23;
	__m512d y24, y25, y26, y27;
	__m512d y28, y29, y30, y31;
		//
	//printf("Starting..."); fflush(stdout);

#ifdef __MIC__
	for( kb = 0; kb < K; kb += K_BLOCK_SIZE ){ int Kb = min( K_BLOCK_SIZE, K - kb );
		for( ib = 0; ib < M; ib += M_BLOCK_SIZE ){ int Mb = min( M_BLOCK_SIZE, M - ib );
			tcopy(Kb, Mb, A + kb*lda + ib, lda, Ab);
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
				//printf("-------> ib = %d, jb = %d, kb = %d\n", ib, jb, kb); fflush(stdout);
				ncopy(Kb, Nb, B + jb*ldb + kb, ldb, Bb);
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
				for (j = 0; j < -Nb; j = j + 1)	
					for (i = 0; i < Mb; i = i + 1){
						AB[j*M_BLOCK_SIZE + i] = 0*C[(j + jb + 0)*ldc + i + ib + 0];
						//printf("before kernel %d %d %f %f\n", i + ib, j + jb, AB[(j + 0)*M_BLOCK_SIZE + i], C[(j + jb + 0)*ldc + i + ib + 0]);
					}

				computetime -= myseconds();
				double* pA = &Ab[0];
				double* pB = &Bb[0];

				for (i = 0; i < Mb - Mb%8; i = i + 8){
					for (j = 0; j < Nb - Nb%16; j = j + 16){

						//printf("i = %d, %d j = %d, %d\n", i, Mb, j, Nb);	
						y00 = _mm512_set1_pd(0.);
						y01 = _mm512_set1_pd(0.);
						y02 = _mm512_set1_pd(0.);
						y03 = _mm512_set1_pd(0.);
						//
						y04 = _mm512_set1_pd(0.);
                                                y05 = _mm512_set1_pd(0.);
                                                y06 = _mm512_set1_pd(0.);
                                                y07 = _mm512_set1_pd(0.);
						//
						y08 = _mm512_set1_pd(0.);
                                                y09 = _mm512_set1_pd(0.);
                                                y10 = _mm512_set1_pd(0.);
                                                y11 = _mm512_set1_pd(0.);
						//
						y12 = _mm512_set1_pd(0.);
                                                y13 = _mm512_set1_pd(0.);
                                                y14 = _mm512_set1_pd(0.);
                                                y15 = _mm512_set1_pd(0.);
						//y14 = LOAD512(&AB[(j + 0)*M_BLOCK_SIZE + i + 4]);
						//y12 = LOAD512(&AB[(j + 1)*M_BLOCK_SIZE + i + 4]);
						//y10 = LOAD512(&AB[(j + 2)*M_BLOCK_SIZE + i + 4]);
						//y08 = LOAD512(&AB[(j + 3)*M_BLOCK_SIZE + i + 4]);
						//
						double* pA = &Ab[i*Kb + 0];
						double* pB = &Bb[j*Kb + 0];
						//
						for (k = 0; k < Kb - Kb%1; k = k + 1)
						{
							//_mm_prefetch((void*) 
							//printf("i = %d j = %d k = %d\n", i, j, k);
							y24 = LOAD512(pA + 0);
							//y25 = LOAD512(pA + 8);
							//
							y31 = _mm512_extload_pd(pB + 0 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
							y29 = _mm512_extload_pd(pB + 8 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
							//printf("y30 : "); print512(y30); printf("\n"); fflush(stdout);
							_mm_prefetch((void*) pA + 512 , _MM_HINT_T0);
							_mm_prefetch((void*) pB + 1024, _MM_HINT_T0);
							//
							y00 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y31, _MM_SWIZ_REG_AAAA), y00);
							y01 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y31, _MM_SWIZ_REG_BBBB), y01);
							y02 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y31, _MM_SWIZ_REG_CCCC), y02);
							y03 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y31, _MM_SWIZ_REG_DDDD), y03);
							//
#if 0
							y04 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 4 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y04);
							y05 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 5 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y05);
							y06 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 6 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y06);
							y07 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 7 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y07);
#else
							y30 = _mm512_extload_pd(pB + 4 , _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
							y04 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y30, _MM_SWIZ_REG_AAAA), y04);
                                                        y05 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y30, _MM_SWIZ_REG_BBBB), y05);
                                                        y06 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y30, _MM_SWIZ_REG_CCCC), y06);
                                                        y07 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y30, _MM_SWIZ_REG_DDDD), y07);
#endif
							//
							y08 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y29, _MM_SWIZ_REG_AAAA), y08);
                                                        y09 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y29, _MM_SWIZ_REG_BBBB), y09);
                                                        y10 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y29, _MM_SWIZ_REG_CCCC), y10);
                                                        y11 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y29, _MM_SWIZ_REG_DDDD), y11);
							//
#if 0
							y12 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y12);
							y13 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 13, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y13);
							y14 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 14, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y14);
							y15 = _mm512_fmadd_pd(y24, _mm512_extload_pd(pB + 15, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, 0), y15);
#else
							y28 = _mm512_extload_pd(pB + 12, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, 0);
							y12 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y28, _MM_SWIZ_REG_AAAA), y12);
                                                        y13 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y28, _MM_SWIZ_REG_BBBB), y13);
                                                        y14 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y28, _MM_SWIZ_REG_CCCC), y14);
                                                        y15 = _mm512_fmadd_pd(y24, _mm512_swizzle_pd(y28, _MM_SWIZ_REG_DDDD), y15);
#endif
							//
							pA += 8;
							pB += 16;
						}
#if 0
						//
						_mm512_store_pd(&AB[(j + 0 )*M_BLOCK_SIZE + i], y00);
						_mm512_store_pd(&AB[(j + 1 )*M_BLOCK_SIZE + i], y01);
						_mm512_store_pd(&AB[(j + 2 )*M_BLOCK_SIZE + i], y02);
						_mm512_store_pd(&AB[(j + 3 )*M_BLOCK_SIZE + i], y03);
						//
						_mm512_store_pd(&AB[(j + 4 )*M_BLOCK_SIZE + i], y04);
                                                _mm512_store_pd(&AB[(j + 5 )*M_BLOCK_SIZE + i], y05);
                                                _mm512_store_pd(&AB[(j + 6 )*M_BLOCK_SIZE + i], y06);
                                                _mm512_store_pd(&AB[(j + 7 )*M_BLOCK_SIZE + i], y07);
						//
						_mm512_store_pd(&AB[(j + 8 )*M_BLOCK_SIZE + i], y08);
                                                _mm512_store_pd(&AB[(j + 9 )*M_BLOCK_SIZE + i], y09);
                                                _mm512_store_pd(&AB[(j + 10)*M_BLOCK_SIZE + i], y10);
                                                _mm512_store_pd(&AB[(j + 11)*M_BLOCK_SIZE + i], y11);
						//
						_mm512_store_pd(&AB[(j + 12)*M_BLOCK_SIZE + i], y12);
                                                _mm512_store_pd(&AB[(j + 13)*M_BLOCK_SIZE + i], y13);
                                                _mm512_store_pd(&AB[(j + 14)*M_BLOCK_SIZE + i], y14);
                                                _mm512_store_pd(&AB[(j + 15)*M_BLOCK_SIZE + i], y15);
#else
						//
						_mm512_store_pd(C + (j + jb + 0)*ldc + i + ib + 0, y00);
						_mm512_store_pd(C + (j + jb + 1)*ldc + i + ib + 0, y01);
						_mm512_store_pd(C + (j + jb + 2)*ldc + i + ib + 0, y02);
						_mm512_store_pd(C + (j + jb + 3)*ldc + i + ib + 0, y03);

                                                _mm512_store_pd(C + (j + jb + 4)*ldc + i + ib + 0, y04);
                                                _mm512_store_pd(C + (j + jb + 5)*ldc + i + ib + 0, y05);
                                                _mm512_store_pd(C + (j + jb + 6)*ldc + i + ib + 0, y06);
                                                _mm512_store_pd(C + (j + jb + 7)*ldc + i + ib + 0, y07);
//
                                                _mm512_store_pd(C + (j + jb + 8)*ldc + i + ib + 0, y08);
                                                _mm512_store_pd(C + (j + jb + 9)*ldc + i + ib + 0, y09);
                                                _mm512_store_pd(C + (j + jb + 10)*ldc + i + ib + 0, y10);
                                                _mm512_store_pd(C + (j + jb + 11)*ldc + i + ib + 0, y11);
//
                                                _mm512_store_pd(C + (j + jb + 12)*ldc + i + ib + 0, y12);
                                                _mm512_store_pd(C + (j + jb + 13)*ldc + i + ib + 0, y13);
                                                _mm512_store_pd(C + (j + jb + 14)*ldc + i + ib + 0, y14);
                                                _mm512_store_pd(C + (j + jb + 15)*ldc + i + ib + 0, y15);
#endif
						//
					}
				}
				computetime += myseconds();
				copytime -= myseconds();
				for (i = 0; i < -Mb; i = i + 1){
					for (j = 0; j < Nb; j = j + 1){
						C[(j + jb + 0)*ldc + i + ib + 0] += alpha*AB[(j + 0)*M_BLOCK_SIZE + i] + beta*C[(j + jb + 0)*ldc + i + ib + 0];
						//printf("after kernel %d %d %f %f\n", i + ib, j + jb, AB[(j + 0)*M_BLOCK_SIZE + i], C[(j + jb + 0)*ldc + i + ib + 0]);
					}
				} // GEBP
				copytime    += myseconds();
			}
		} //
	}
#endif
	//printf("copy time = %f, compute time = %f, %f GFlops\n", copytime, computetime, 2.*M*N*K/computetime/1e9);
}


void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
#pragma offload target(mic) \
	in(A[0:K*lda]:align(64)),\
	in(B[0:N*ldb]:align(64)),\
	out(C[0:N*ldc]:align(64)) \
	in(M)\
	in(N)\
	in(K)\
	in(lda)\
	in(ldb)\
	in(ldc)\
	in(alpha)\
	in(beta)
	{
		dgemm_knc(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}	


