#include <cmath>
#include <cstdio>
#include <algorithm>
#include <unistd.h>

#include "defs.h"

//rdtsc acknowledgement:
//http://stackoverflow.com/questions/9887839/clock-cycle-count-wth-gcc
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

extern "C" void upwind3( 
    const  real uin[], const  real vin[], const  real win[],
    const  real factor, real uout[],   real vout[],   real wout[]);

int main()
{
    real * velocity[3];

    for(int c = 0; c < 3; c++)
	velocity[c] = new real [INPUTVOLUME ];

    const real h = 1. / OUTPUTSLICE;
    
    for (int iz = 0; iz < INPUTSTRIDE; iz++)
	for (int iy = 0; iy < INPUTSTRIDE; iy++)
	    for (int ix=0; ix < INPUTSTRIDE; ix++)
	    {
		const real x = (ix - 2) * h;
		const real y = (iy - 2) * h;
		const real z = (iz - 2) * h;
		const int entry = ix + INPUTSTRIDE * (iy + INPUTSTRIDE * iz);
		
		velocity[0][entry] = x;
		velocity[1][entry] = x;
		velocity[2][entry] = x;
	    }	
    
    real * rhs[3];
    for(int c = 0; c < 3; ++c)
	rhs[c] = new real [OUTPUTVOLUME];

    const int npasses = 1000;
    size_t timings[npasses];
       
    for(int p = 0; p < npasses; ++p)
    {
	usleep(100);
	
	const size_t startc = rdtsc();

	const int bias =  2 * (INPUTSLICE + INPUTSTRIDE + 1);
	upwind3(velocity[0] + bias, velocity[1] + bias, velocity[2] + bias,  -1 / (6 * h), rhs[0], rhs[1], rhs[2]);

	timings[p] = rdtsc() - startc;
    }
    
    std::sort(timings, timings + npasses);
    
    const int p01 = (int)(0.01 * npasses);
    const int p99 = (int)(0.99 * npasses);
    
    printf("PERFORMANCE WITHIN %.1f - %.1f FLOP/CYCLE (98%% CONFIDENCE)\n",
	   144 * OUTPUTVOLUME * 1. / timings[p99],
	   144 * OUTPUTVOLUME * 1. / timings[p01]);
    
    for(int c = 0; c < 3; ++c)
	delete [] rhs[c];
    
    for(int c = 0; c < 3; ++c)
	delete [] velocity[c];

    return 0;
}
