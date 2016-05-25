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

define(CPPKERNELS, `
#include <cinttypes>
#include <immintrin.h>
	 
extern "C" void bitshuffle_8x32(uint32_t * const data)
{
	LUNROLL(i, 0, 7, `
	uint32_t TMP(src, i) = data[i];')
	
	for(int w = 0; w < 8; ++w)
	{
		uint32_t code = 0;
		LUNROLL(i, 0, 7, `
		LUNROLL(B, 0, 3, `
		code |= ((TMP(src, i) & (1 << (B + 4 * w))) != 0) << eval(i + 8 * B);')')

		data[w] = code;
	}
}

extern "C" void bitshuffle_32x8(uint32_t * const data)
{
	LUNROLL(i, 0, 7, `
	uint32_t TMP(src, i) = data[i];')
	
	for(int w = 0; w < 8; ++w)
	{
		uint32_t code = 0;
		
		LUNROLL(i, 0, 7, `
		LUNROLL(B, 0, 3, `
		code |= ((TMP(src, i) & (1 << (w + (8 * B)))) != 0) << eval(B + 4 * i);')')
		
		data[w] = code;
	}
}')

define(ISPCKERNELS, `
	 
export void bitshuffle_8x32(uniform unsigned int data[])
{
	LUNROLL(i, 0, 7, `
	uniform unsigned int TMP(src, i) = data[i];')
	
	foreach(w = 0 ... 8)
	{
		unsigned int code = 0;
		LUNROLL(i, 0, 7, `
		LUNROLL(B, 0, 3, `
		{
		const unsigned int bitval = (TMP(src, i) & (1 << (B + 4 * w))) > 0;
		code |= bitval << eval(i + 8 * B);
		}')')

		data[w] = code;
	}
}

export void bitshuffle_32x8(uniform unsigned int data[])
{
	LUNROLL(i, 0, 7, `
	uniform unsigned int TMP(src, i) = data[i];')
	
	foreach(w = 0 ... 8)
	{
		unsigned int code = 0;
		
		LUNROLL(i, 0, 7, `
		LUNROLL(B, 0, 3, `
		{
		const unsigned int bitval = (TMP(src, i) & (1 << (w + (8 * B)))) > 0;
		code |= bitval << eval(B + 4 * i);
		}')')
				
		data[w] = code;
	}
}')

divert(0)

ifelse(KERNELS, ISPC, ISPCKERNELS, CPPKERNELS)
