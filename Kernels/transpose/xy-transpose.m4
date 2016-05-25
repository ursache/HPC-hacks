#/*
#* Created and authored by Diego Rossinelli on 2015-11-25.
#* Copyright 2015. All rights reserved.
#*
#* Users are NOT authorized
#* to employ the present software for their own publications
#* before getting a written permission from the author of this file.
#*/


divert(-1)
include(common.m4)
define(BS,256)

define(CPPKERNELS, `
void transpose_block(float block[BS][BS])
{
	LUNROLL(iy, 0, eval(BS - 1), `dnl
	ifelse(eval(iy + 1 < BS), 1, `dnl
	LUNROLL(ix, eval(iy + 1), eval(BS - 1), `
	{
		const float tmp = block[iy][ix];
		block[iy][ix] = block[ix][iy];
		block[ix][iy] = tmp;
	}')')')
}

void load_block(const float * const data, const int xstride, float block[BS][BS])
{
	for(int iy = 0; iy < BS; ++iy)
	{
		const int srcbase = xstride * iy;
		LUNROLL(ix, 0, eval(BS - 1), `
		block[iy][ix] = data[ix + srcbase];')
	}
}

void store_block(const float block[BS][BS], const int xstride, float * data)
{
	for(int iy = 0; iy < BS; ++iy)
	{
		const int dstbase = xstride * iy;
		LUNROLL(ix, 0, eval(BS - 1), `
		data[ix + dstbase] = block[iy][ix];')
	}
}
') #end of CPPKERNELS

define(AVXKERNELS, `
#include <immintrin.h>

#define TR8x8(v0, v1, v2, v3, v4, v5, v6, v7)	\
    {								\
    __m256 __t0 = _mm256_unpacklo_ps(v0, v1);			\
    __m256 __t1 = _mm256_unpackhi_ps(v0, v1);			\
    __m256 __t2 = _mm256_unpacklo_ps(v2, v3);			\
    __m256 __t3 = _mm256_unpackhi_ps(v2, v3);			\
    __m256 __t4 = _mm256_unpacklo_ps(v4, v5);			\
    __m256 __t5 = _mm256_unpackhi_ps(v4, v5);			\
    __m256 __t6 = _mm256_unpacklo_ps(v6, v7);			\
    __m256 __t7 = _mm256_unpackhi_ps(v6, v7);			\
									\
    __m256 __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));		\
    __m256 __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));		\
    __m256 __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));		\
    __m256 __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));		\
    __m256 __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));		\
    __m256 __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));		\
    __m256 __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));		\
    __m256 __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));		\
									\
    v0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);			\
    v1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);			\
    v2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);			\
    v3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);			\
    v4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);			\
    v5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);			\
    v6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);			\
    v7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);			\
    }

void transpose_tile(float block[BS][BS], const int base)
{
	LUNROLL(i, 0, 7, `dnl
	__m256 TMP(row, i) = _mm256_load_ps(&block[i + base][base]);')

	TR8x8(row_0, row_1, row_2, row_3, row_4, row_5, row_6, row_7);

	LUNROLL(i, 0, 7, `dnl
	_mm256_store_ps(&block[i + base][base], TMP(row, i));')
}

void transpose_2tiles(float block[BS][BS], const int xbase, const int ybase)
{
	LUNROLL(i, 0, 7, `dnl
	__m256 TMP(row, i) = _mm256_load_ps(&block[i + ybase][xbase]);')

	LUNROLL(i, 0, 7, `dnl
	__m256 TMP(col, i) = _mm256_load_ps(&block[i + xbase][ybase]);')

	TR8x8(row_0, row_1, row_2, row_3, row_4, row_5, row_6, row_7);
	TR8x8(col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7);
		
	LUNROLL(i, 0, 7, `dnl
	_mm256_store_ps(&block[i + xbase][ybase], TMP(row, i));')

	LUNROLL(i, 0, 7, `dnl
	_mm256_store_ps(&block[i + ybase][xbase], TMP(col, i));')
}

void transpose_block(float block[BS][BS])
{
	for(int yb = 0; yb < (BS / 8); ++yb)
	{
		transpose_tile(block, 8 * yb);

		for(int xb = yb + 1; xb < (BS / 8); ++xb)
			transpose_2tiles(block, 8 * xb, 8 * yb);
	}
}

void load_block(const float * const data, const int xstride, float block[BS][BS])
{
	for(int iy = 0; iy < BS; ++iy)
	{
		const int srcbase = xstride * iy;
		LUNROLL(xbase, 0, eval(BS / 8 - 1), `
		_mm256_store_ps(&block[iy][eval(xbase * 8)], _mm256_loadu_ps(&data[eval(xbase * 8) + srcbase]));')
	}
}

void store_block(const float block[BS][BS], const int xstride, float * data)
{
	for(int iy = 0; iy < BS; ++iy)
	{
		const int dstbase = xstride * iy;
		LUNROLL(xbase, 0, eval(BS / 8 - 1), `
		_mm256_storeu_ps(&data[eval(xbase * 8) + dstbase], _mm256_load_ps(&block[iy][eval(xbase * 8)]));')
	}
}
') #end of AVX KERNELS
divert(0)
ifelse(KERNELS, avx, AVXKERNELS, CPPKERNELS)

extern "C" void xy_transpose(const float * const src, float * const dst, const int nx, const int ny, const int nz)
{
	const int xblocks = (nx + eval(BS - 1)) / BS;	
	const int yblocks = (ny + eval(BS - 1)) / BS;
		
	const int nblocks = xblocks * yblocks * nz;

#pragma omp parallel
	{

	float block[BS][BS] __attribute__ ((aligned (32)));
#pragma omp for
	for(int b = 0; b < nblocks; ++b)
	{
		const int bx = b % xblocks;
		const int by = (b / xblocks) % yblocks;
		const int iz = b / (xblocks * yblocks);

		const float * const srcslice = src + nx * ny * iz;
		float * const dstslice = dst + nx * ny * iz;
		
		const int xs = bx * BS;
		const int ys = by * BS;
		
		const int xe = xs + BS > nx ? nx : xs + BS;
		const int ye = ys + BS > ny ? ny : ys + BS;

		const int nxdst = xe - xs;
		const int nydst = ye - ys;

		const bool fullblock = nxdst * nydst == BS * BS;

		if (fullblock)
		   load_block(srcslice + xs + nx * ys, nx, block);
		else
		    for(int dy = 0; dy < nydst; ++dy)
		    	    for(int dx = 0; dx < nxdst; ++dx)
			    	    block[dy][dx] = srcslice[dx + xs + nx * (ys + dy)];

		transpose_block(block);

		if (fullblock)
		   store_block(block, ny, dstslice + ys + ny * xs);
		else
		    for(int dx = 0; dx < nxdst; ++dx)
		    	    for(int dy = 0; dy < nydst; ++dy)
			    	    dstslice[dy + ys + ny * (xs + dx)] = block[dx][dy];
	}
	}
}
