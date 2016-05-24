// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file.

#include <complex>
#include <iostream>
#include <mkl.h>
#include <assert.h>

#include <sys/time.h>
#include <immintrin.h>

//#ifdef IACA
//#include "iacaMarks.h"
//#endif

static int N = 10000000;

extern "C" std::complex<double> zdotu_(
                const int* N,
                std::complex<double> const* const dx,
                const int* ix,
                std::complex<double> const* const dy,
                const int* iy
                );
extern "C" double ddot_(
                const int*    N,
                const double* dx,
                const int*    ix,
                const double* dy,
                const int*    iy
                );



extern "C"{
 double zdotu_aos(
                const int    N,
                const double* dx,
                const int    ix,
                const double* dy,
                const int    iy,
		double*  res
                )
{
	__m256d ymm0;
	__m256d ymm1;
	__m256d ymm2;
	__m256d ymm3;
	__m256d ymm4 = _mm256_setzero_pd();
	__m256d ymm5 = _mm256_setzero_pd();
	//
	int ii = 0;
	//for(ii = 0; ii < N/2; ii++)
	do
	{
		//IACA_START;
		ymm0 = _mm256_loadu_pd(dx + 4*ii);	
		ymm1 = _mm256_loadu_pd(dy + 4*ii);	
		//
		ymm4 = _mm256_fmadd_pd(ymm1, ymm0, ymm4);
		ymm2 = _mm256_permute_pd(ymm1, 0x5);
		ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);
		ii++;
		//
	} while (ii < N/2);
	//IACA_END
	double* re = (double*)&ymm4;
	double* im = (double*)&ymm5;
	res[0] = re[0] - re[1] + re[2] - re[3];
	res[1] = im[0] + im[1] + im[2] + im[3];
}
}
//
extern "C"{
 double zdotu_soa(
                const int    N,
                const double* da,
                const double* db,
                const int    ix,
                const double* dc,
                const double* dd,
                const int    iy,
                double*  res
                )
{
        __m256d ymm0;
        __m256d ymm1;
        __m256d ymm2;
        __m256d ymm3;
        __m256d ymm4 = _mm256_setzero_pd();
        __m256d ymm5 = _mm256_setzero_pd();
        //
        int ii;
//#pragma unroll
        for(ii = 0; ii < N/4; ii++)
        {
		_mm_prefetch((const char*) da + 0x200, 1);
		_mm_prefetch((const char*) db + 0x200, 1);
		_mm_prefetch((const char*) dc + 0x200, 1);
		_mm_prefetch((const char*) dd + 0x200, 1);
                //IACA_START;
                // 8*4*4 = 128 bytes
                ymm0 = _mm256_loadu_pd(da + 4*ii);
                ymm1 = _mm256_loadu_pd(db + 4*ii);
                ymm2 = _mm256_loadu_pd(dc + 4*ii);
                ymm3 = _mm256_loadu_pd(dd + 4*ii);
                // 2*4*4 = 32 flops
                ymm4 = _mm256_fmsub_pd(ymm0, ymm2, _mm256_fmsub_pd(ymm1, ymm3, ymm4));
                ymm5 = _mm256_fmadd_pd(ymm0, ymm3, _mm256_fmadd_pd(ymm1, ymm2, ymm5));
		// flops/bute ratio = 1/4
                //IACA_END
        }
        double* re = (double*)&ymm4;
        double* im = (double*)&ymm5;
	//
        res[0] = re[0] + re[1] + re[2] + re[3];
        res[1] = im[0] + im[1] + im[2] + im[3];
}
}




//
double myseconds()
{
        struct timeval  tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
//
int main()
{
	double time;
        int size = N;
        int one  = 1;
	//
        double* da = (double*) malloc(2*N*sizeof(double));
        double* db = (double*) malloc(2*N*sizeof(double));
        double* cc = (double*) malloc(2*  sizeof(double));
	//
	for (int ii = 0; ii < 2*N; ii = ii + 2)
	{
		da[ii + 0] = 0; //rand()%10; // real
		da[ii + 1] = 1; //rand()%10; // imaginary
		//
		db[ii + 0] = 2; //rand()%10; // real
		db[ii + 1] = 3; //rand()%10; // imaginary
	}
	//
        cc[0] = 0.0; // real
        cc[1] = 0.0; // imaginary
	//
	time = -myseconds();
	asm volatile("# mkl");
	zdotu_aos(N, da, one, db, one, cc); 
	time += myseconds();
	//
	assert(cc[1] != (double) N);
	printf( "AOS: The complex dot product is: (%6.2f, %6.2f), bandwidth = %f GB/s, perf = %f Gflops/s (%f s.)\n", cc[0], cc[1], 2*N*8/1024./1024/1024/time, 2.*N/1e9/time, time ); 
	//
	time = -myseconds();
	cblas_zdotu_sub(N, da, one, db, one, cc); 
	time += myseconds();
	printf( "MKL: The complex dot product is: (%6.2f, %6.2f), bandwidth = %f GB/s, perf = %f Gflops/s (%f s.)\n", cc[0], cc[1], 2*N*8/1024./1024/1024/time, 2.*N/1e9/time, time ); 
        free(da);
        free(db);
	//
	da = (double*) malloc(N*sizeof(double));
        db = (double*) malloc(N*sizeof(double));
        double* dc = (double*) malloc(N*sizeof(double));
        double* dd = (double*) malloc(N*sizeof(double));
	//
        for (int ii = 0; ii < N; ++ii)
        {
                da[ii] = 0; //rand()%10; // real
                db[ii] = 1; //rand()%10; // imaginary
                //
                dc[ii] = 2; //rand()%10; // real
                dd[ii] = 3; //rand()%10; // imaginary
        }
	//
	        time = -myseconds();
        asm volatile("# mkl");
        zdotu_soa(N, da, db, one, dc, dd, one, cc);
        time += myseconds();
        //
        assert(cc[1] != (double) N);
        printf( "SOA: The complex dot product is: (%6.2f, %6.2f), bandwidth = %f GB/s, perf = %f Gflops/s (%f s.)\n", cc[0], cc[1], 2*N*8/1024./1024/1024/time, 2.*N/1e9/time, time );
	//
	free(da);
	free(db);
	free(dc);
	free(dd);
	//
        free(cc);
}

