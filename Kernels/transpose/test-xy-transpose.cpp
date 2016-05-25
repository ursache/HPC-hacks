/*
* test-xy-transpose.cpp 
* Created and authored by Diego Rossinelli on 2015-11-25.
* Copyright 2015. All rights reserved.
*
* Users are NOT authorized
* to employ the present software for their own publications
* before getting a written permission from the author of this file.
*/


#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cinttypes>

#include <omp.h>

extern "C" void xy_transpose(const float * const src, float * const dst, const int nx, const int ny, const int nz);

uint64_t rdtsc() 
{
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void xy_gold(const float * src, float * const dst, const int nx, const int ny)
{
    for(int iy = 0; iy < ny; ++iy)
	for(int ix = 0; ix < nx; ++ix)
	    dst[iy + ny * ix] = src[ix + nx * iy];
}

void test1()
{
    const int nx = 2304;
    const int ny = 2304;

    float * ic = new float[nx * ny];
    float * ref = new float[nx * ny];
    float * res = new float[nx * ny];

    for(int iy = 0; iy < ny; ++iy)
	for(int ix = 0; ix < nx; ++ix)
	    ic[ix + nx * iy] = ix + nx * iy;

    memcpy(res, ic, sizeof(float) * nx * ny);
    memcpy(ref, ic, sizeof(float) * nx * ny);

    const int ntimes = 100;

    const size_t t0 = rdtsc();

    for(int i = 0; i < ntimes; ++i)
	xy_gold(ic, ref, nx, ny);

    const size_t t1 = rdtsc();
    const double tstart = omp_get_wtime();
    
    for(int i = 0; i < ntimes; ++i)
	xy_transpose(ic, res, nx, ny, 1);

    const double tend = omp_get_wtime();
    const size_t t2 = rdtsc();

    if (memcmp(ref, res, sizeof(float) * nx * ny) != 0)
    {
	printf("\x1b[41mBAD!\x1b[0m\n");
	abort();
    }

    delete [] ic;
    delete [] ref;
    delete [] res;

    const double tts = (tend - tstart) / ntimes;
    printf("XY-TRANSPOSE: %.2f MB\n", 2 * nx * ny * sizeof(float) / 1024. / 1024.);
    printf("XY-TRANSPOSE: gold cycles: %.3e\n", (size_t)(t1 - t0) / (double)ntimes);
    printf("XY-TRANSPOSE: cycles: %.3e MEM BW: \x1b[91m%.2f GB/s, %.2f B/c\x1b[0m\n",
	   (size_t)(t2 - t1) / (double)ntimes,
	   2 * nx * ny * sizeof(float) / tts / 1024. / 1024. / 1024.,
	   2 * nx * ny * sizeof(float)  / (double)(t2 - t1) * ntimes );
    printf("XY-TRANSPOSE: improvement: \x1b[92m%.2fX\x1b[0m\n", (t1 - t0) / (double)(t2 - t1));
    
}

int main()
{
    test1();
}
