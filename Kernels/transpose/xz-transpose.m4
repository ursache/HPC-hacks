#/*
#* Created and authored by Diego Rossinelli on 2015-11-25.
#* Copyright 2015. All rights reserved.
#*
#* Users are NOT authorized
#* to employ the present software for their own publications
#* before getting a written permission from the author of this file.
#*/



divert(-1)
define(`forloop',
       `pushdef(`$1', `$2')_forloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', incr($1))_forloop(`$1', `$2', `$3', `$4')')')

define(`forrloop',
       `pushdef(`$1', `$2')_forrloop(`$1', `$2', `$3', `$4')popdef(`$1')')

define(`_forrloop',
       `$4`'ifelse($1, `$3', ,
		   `define(`$1', decr($1))_forrloop(`$1', `$2', `$3', `$4')')')

USAGE LUNROLL
$1 iteration variable
$2 iteration start
$3 iteration end
$4 body

define(LUNROLL, `forloop($1, $2, $3,`$4')')
define(RLUNROLL, `forrloop($1, $2, $3, `$4')')
define(`TMP', $1_$2)

define(TR8x8, `
{
   __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
 
   __t0 = _mm256_unpacklo_ps(row_0, row_1);
   __t1 = _mm256_unpackhi_ps(row_0, row_1);
   __t2 = _mm256_unpacklo_ps(row_2, row_3);
   __t3 = _mm256_unpackhi_ps(row_2, row_3);
   __t4 = _mm256_unpacklo_ps(row_4, row_5);
   __t5 = _mm256_unpackhi_ps(row_4, row_5);
   __t6 = _mm256_unpacklo_ps(row_6, row_7);
   __t7 = _mm256_unpackhi_ps(row_6, row_7);

    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
   __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
   __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
   __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
   __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
   __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
   __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
   __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
   __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  
   row_0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
   row_1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
   row_2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
   row_3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
   row_4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
   row_5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
   row_6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
   row_7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}')

divert(0)
#include <immintrin.h>
#include <cassert>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
divert(-1)
define(BS, 8)
define(`XZ_TR1', `
{
//single transpose with x=$2 y=$1 z=$2
LUNROLL(c, 0, 7, `__m256 TMP(row, c) = _mm256_load_ps(data + eval($2 + BS * ($1 + BS * ($2 + c))));
')
TR8x8

LUNROLL(c, 0, 7, `_mm256_store_ps(data + eval($2 + BS * ($1 + BS * ($2 + c))), TMP(row, c));
')dnl
}')

define(FUNCNAME, $1_$2x$2x$2)

#define(`XZ_TR2', `
#{
#//double transpose with x=$1 y=$2 z=$3
#}')
divert(0)
extern "C" void FUNCNAME(xz_transpose, BS)(float * data)
{
	LUNROLL(iy, 0, eval(BS - 1), `LUNROLL(basez, 0, eval(BS / 8 - 1), ` define(iz, `eval(basez * 8)')
	XZ_TR1(iy, iz) LUNROLL(basex, basez, eval(BS / 8 - 1), ` define(ix, `eval(basex * 8)')
	ifelse(eval(basex > basez),1, `XZ_TR2(ix, iy, iz)')')')')dnl
}

void FUNCNAME(load, BS)(const float * const src, const int xsize, const int ysize, float block[BS][BS][BS])
{
	LUNROLL(dz, 0, eval(BS - 1),`
	{
	const float * const slice = src + xsize * ysize * dz;
	LUNROLL(dy, 0, eval(BS - 1),`LUNROLL(dx, 0, eval(BS / 8 - 1),`
	_mm256_store_ps(&block[dz][dy][dx], _mm256_loadu_ps(slice + dy * xsize));')')}')
}

void FUNCNAME(store, BS)(float block[BS][BS][BS], float * const dst, const int xsize, const int ysize)
{
	LUNROLL(dz, 0, eval(BS - 1),`
	{
	float * const slice = dst + xsize * ysize * dz;
	LUNROLL(dy, 0, eval(BS - 1),`LUNROLL(dx, 0, eval(BS / 8 - 1),`
	_mm256_storeu_ps(slice + dy * xsize, _mm256_load_ps(&block[dz][dy][dx]));')')}')
}

divert(-1)
popdef(`ix')
popdef(`iy')
popdef(`iz')
divert(0)
extern "C" void xz_transpose(
    const float * const src,
    const int xsize,
    const int ysize,
    const int zsize,
    float * const dst)
    {
	const int xcubes = (xsize + BS - 1) / BS;
	const int ycubes = (ysize + BS - 1) / BS;
    	const int zcubes = (zsize + BS - 1) / BS;
    	const int ncubes = xcubes * ycubes * zcubes;

	const int A = (int)(xsize >= zsize);
	const int B = 1 - A;

#pragma omp parallel
    	{
		float cube[BS][BS][BS] __attribute__ ((aligned (32)));

#pragma omp for
		for(int i = 0; i < ncubes; ++i)
		{
			const int zcubeA = i % zcubes;
			const int xcubeA = (i / zcubes) % xcubes;
			const int xcubeB = i % xcubes;
			const int zcubeB = (i / xcubes) % zcubes;
			
			const int xcube = A * xcubeA + B * xcubeB;
			const int zcube = A * zcubeA + B * zcubeB;
			
			const int ycube = i / (xcubes * zcubes);
			
			assert(xcube < xcubes);
			assert(ycube < ycubes);
			assert(zcube < zcubes);

			const int xstart = xcube * BS;
			const int ystart = ycube * BS;
			const int zstart = zcube * BS;

			const int xcount = MIN((int)BS, xsize - xstart);
			const int ycount = MIN((int)BS, ysize - ystart);
			const int zcount = MIN((int)BS, zsize - zstart);

			const bool fullblock = xcount * ycount * zcount == eval(BS * BS * BS);

			const float * const srcbase = src + (size_t)xstart + (size_t)xsize * ((size_t)ystart + (size_t)ysize * (size_t)zstart);

			if (fullblock)
		   	   load_8x8x8(srcbase, xsize, ysize, cube);
			else
				for(int iz = 0; iz < zcount; ++iz)
				for(int iy = 0; iy < ycount; ++iy)
				for(int ix = 0; ix < xcount; ++ix)
					cube[iz][iy][ix] = srcbase[ix + xsize * (iy + ysize * iz)];

			xz_transpose_8x8x8((float *)cube);

			float * const dstbase = dst + (size_t)zstart + (size_t)zsize * ((size_t)ystart + (size_t)ysize * (size_t)xstart);

			if (fullblock)
		   	   store_8x8x8(cube, dstbase, zsize, ysize);
			else
				for(int ix = 0; ix < xcount; ++ix)
				for(int iy = 0; iy < ycount; ++iy)
				for(int iz = 0; iz < zcount; ++iz)
					dstbase[iz + zsize * (iy + ysize * ix)] = cube[ix][iy][iz];
		}
	}
}


