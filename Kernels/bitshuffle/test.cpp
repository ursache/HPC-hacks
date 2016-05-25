/*
* Created and authored by Diego Rossinelli on 2015-11-25.
* Copyright 2015. All rights reserved.
*
* Users are NOT authorized
* to employ the present software for their own publications
* before getting a written permission from the author of this file.
*/

#include <cstdlib>
#include <cstdio>
#include <cinttypes>
#include <cassert>

#include "bitshuffle.h"

uint64_t rdtsc()
{
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

extern "C" void gold_forward(uint32_t * const seq)
{
    uint8_t shuf[32];
	
    for(int b = 0; b < 32; ++b)
    {
	shuf[b] = 0;
	for(int i = 0; i < 8; ++i)
	{
	    const int val = (seq[i] & (1 << b)) != 0;
	    shuf[b] |= val << i;
	}
    }

    for(int c = 0; c < 8; ++c)
	seq[c] = *(c + (uint32_t *)shuf);
}

int test_forward()
{
    printf("TEST FORWARD\n");
    
    const int ntimes = 10000;
    
    uint32_t seq[8], seqref[8];
    for(int i = 0; i < 8; ++i)
	seq[i] = seqref[i] = rand();

    const size_t t0 = rdtsc();
    
    for(int i = 0; i < ntimes; ++i)
	gold_forward(seqref);
		
    const size_t t1 = rdtsc();

    for(int i = 0; i < ntimes; ++i)
	bitshuffle_8x32(seq);

    const size_t t2 = rdtsc();
	
    for(int b = 0; b < 32; ++b)
    {
	uint8_t res = *(b + (uint8_t *)seq);
	uint8_t ref = *(b + (uint8_t *)seqref);
	
	//printf("shuffled res %0d: 0x%02x vs ref 0x%02x\n", b, res, ref);
	
	if (res != ref)
	{
	    printf("\x1b[41mBAD!\x1b[0m\n");
	    abort();
	}
    }

    printf("FORWARD: gold cycles: %zd\n", (size_t)(t1 - t0) / ntimes);
    printf("FORWARD: cycles: %zd\n", (size_t)(t2 - t1) / ntimes);
    printf("FORWARD: improvement: %.2fX\n", (t1 - t0) / (double)(t2 - t1));
}

void test_backward()
{
    printf("TEST BACKWARD\n");
     
    uint32_t seq[8], seqref[8];
    for(int i = 0; i < 8; ++i)
	seq[i] = seqref[i] = rand();

    printf("first test:\n");

    bitshuffle_8x32(seq);
    
    bitshuffle_32x8(seq);
	
    for(int b = 0; b < 32; ++b)
    {
	uint8_t res = *(b + (uint8_t *)seq);
	uint8_t ref = *(b + (uint8_t *)seqref);	

	//printf("shuffled res %0d: 0x%02x vs ref 0x%02x\n", b, res, ref);
	if (res != ref)
	{
	    printf("\x1b[41mBAD!\x1b[0m\n");
	    abort();
	}
    }

    const int ntimes = 10000;

    const size_t t0 = rdtsc();
    
    for(int i = 0; i < ntimes; ++i)
	bitshuffle_32x8(seq);
    
    const size_t t1 = rdtsc();

    printf("BACKWARD: cycles: %zd\n", (size_t)(t1 - t0) / ntimes);
}

int main()
{
    test_forward();
    
    test_backward();

    return 0;
}
