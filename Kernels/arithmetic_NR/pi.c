// base code from https://techblog.lankes.org/2014/06/16/AVX-isnt-always-faster-than-SEE/
// author gilles.foureste@epfl.ch

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>

#include "simd_math.h"

double width;
double sum;
int num_rects = 1000000000;

//#define PI 3.14159265358979
#define PI 3.1415926535897932


double myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}



double calcPi_double()
{
        double x;
        int i;
	double sum = 0.;
	//
        for (i = 0; i < num_rects; i++) 
	{
                x = (i + 0.5)*width;
                sum += 4.0/(1.0 + x*x);
        }
	return width*sum;
}

double calcPi_float()
{
        float x;
        int i;
	//
        for (i = 0; i < num_rects; i++)
        {
                x = (i + 0.5)*width;
                sum += 4.0/(1.0 + x*x);
        }
        return width*sum;
}
//
//
//
double calcPi_simd(void)
{
        double x;
        int i;
        double width = 1./(double) num_rects;
        //
        __m256d __width = _mm256_set1_pd(width);
        //
        __m256d __half  = _mm256_set1_pd(0.5);
        __m256d __four  = _mm256_set1_pd(4.0);
        __m256d __one   = _mm256_set1_pd(1.0);
        //
        __m256d __sum   = _mm256_set1_pd(0.0);
        __m256d __i = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);

        for (i = 0; i < num_rects; i += 4)
        {
                __m256d __x = __i *__width;
                __m256d __y = _mm256_div_pd(__one, __one + __x*__x);
                __sum      += __four * __y; 
                __i         = __i + __four;
        }
        double sum;
        //
        sum  = ((double*) &__sum)[0];
        sum += ((double*) &__sum)[1];
        sum += ((double*) &__sum)[2];
        sum += ((double*) &__sum)[3];
        //
        return width*sum;
}
//
//
//
double calcPi_simd_rcp(void)
{
        double x;
        int i;
        double width = 1./(double) num_rects;
        //
        __m256d __width = _mm256_set1_pd(width);
        //
        __m256d __half  = _mm256_set1_pd(0.5);
        __m256d __four  = _mm256_set1_pd(4.0);
        __m256d __one   = _mm256_set1_pd(1.0);
        //
        __m256d __sum   = _mm256_set1_pd(0.0);
        __m256d __i = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);

        for (i = 0; i < num_rects; i += 4)
        {
                __m256d __x = __i *__width;
                __m256d __y = RCP(__one + __x*__x);
                __sum      += __four * __y; // (__one + __x*__x);
                __i         = __i + __four;
        }
        double sum;
        //
        sum  = ((double*) &__sum)[0];
        sum += ((double*) &__sum)[1];
        sum += ((double*) &__sum)[2];
        sum += ((double*) &__sum)[3];
        //
        return width*sum;
}
//
//
//
double calcPi_simd_1nr(void)
{
        double x;
        int i;
	double width = 1./(double) num_rects;
	//
	__m256d __width = _mm256_set1_pd(width);
	//
	__m256d __half  = _mm256_set1_pd(0.5);
	__m256d __four  = _mm256_set1_pd(4.0);
	__m256d __one   = _mm256_set1_pd(1.0);
	//
	__m256d __sum   = _mm256_set1_pd(0.0);
	__m256d __i = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);

        for (i = 0; i < num_rects; i += 4) 
	{
                __m256d __x = __i *__width;
		__m256d __y = RCP_1NR(__one + __x*__x);
                __sum += __four * __y; // (__one + __x*__x);
		__i = __i + __four;
        }
	double sum;
	//
	sum  = ((double*) &__sum)[0];
	sum += ((double*) &__sum)[1];
	sum += ((double*) &__sum)[2];
	sum += ((double*) &__sum)[3];
	//
	return width*sum;
}

double calcPi_simd_2nr(void)
{
        double x;
        int i;
        double width = 1./(double) num_rects;
        //
        __m256d __width = _mm256_set1_pd(width);
        //
        __m256d __half  = _mm256_set1_pd(0.5);
        __m256d __four  = _mm256_set1_pd(4.0);
        __m256d __one   = _mm256_set1_pd(1.0);
        //
        __m256d __sum   = _mm256_set1_pd(0.0);
        __m256d __i = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);

        for (i = 0; i < num_rects; i += 4)
        {
                __m256d __x = __i *__width;
                //__m256d __y = _mm256_div_pd(__one, __one + __x*__x);
                __m256d __y = RCP_2NR(__one + __x*__x);
                __sum += __four * __y; // (__one + __x*__x);
                __i = __i + __four;
        }
        double sum;
        //
        sum  = ((double*) &__sum)[0];
        sum += ((double*) &__sum)[1];
        sum += ((double*) &__sum)[2];
        sum += ((double*) &__sum)[3];
        //
        return width*sum;
}



int main(int argc, char **argv)
{
        double time;

        if (argc > 1)
                num_rects = atoi(argv[1]);
        if (num_rects < 100)
                num_rects = 1000000;
        printf("\nnum_rects = %d\n", (int)num_rects);
	printf("true   PI = %.15f\n", PI);
	//
	width = 1.0 / (double)num_rects;
	//
        //width = 1.0 / (double)num_rects;
	//
	// single precision
	//
	time   = -myseconds();
        sum = calcPi_float();
        time  +=  myseconds();
        //
	printf("float  PI = %.15f, error = %g, ", sum, fabs((sum - PI)/PI));
	printf("Time : %lf sec\n", time);
	//
	// double precision	
	//
	time   = -myseconds();
        sum  = calcPi_double();
	time  +=  myseconds();
	//
        printf("double PI = %.15f, error = %lg, ", sum, fabs((sum - PI)/PI));
        printf("Time : %lf sec\n", time); 
        //
        // simd double precision
        //
        time  = -myseconds();
        sum = calcPi_simd();
        time +=  myseconds();
        //
        printf("SIMD   PI = %.15f, error = %g, ", sum, fabs((sum - PI)/PI));
        printf("Time : %lf sec\n", time);
        //
        // simd double precision
        //
        time  = -myseconds();
        sum = calcPi_simd_rcp();
        time +=  myseconds();
        //
        printf("RCP    PI = %.15f, error = %g, ", sum, fabs((sum - PI)/PI));
        printf("Time : %lf sec\n", time);
	//
	// simd double precision
	//
	time  = -myseconds();
        sum = calcPi_simd_1nr();
	time +=  myseconds();
	//
        printf("1 NR   PI = %.15f, error = %g, ", sum, fabs((sum - PI)/PI));
        printf("Time : %lf sec\n", time); 
	//
	//
	//
        time  = -myseconds();
        sum = calcPi_simd_2nr();
        time +=  myseconds();
        //
        printf("2 NR   PI = %.15f, error = %g, ", sum, fabs((sum - PI)/PI));
        printf("Time : %lf sec\n", time);
        //
        //
        //	
        return 0;
}
