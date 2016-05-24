// 
// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file
//

#pragma once

#define MIN(a,b)  (((a<b)?a:b))


static inline unsigned long long cycles() 
{
  unsigned long long u;
  //asm volatile ("rdtsc;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"(u)::"%rdx");
  asm volatile ("rdtscp;shlq $32,%%rdx;orq %%rdx,%%rax;movq %%rax,%0":"=q"(u)::"%rax", "%rdx", "rcx");
  return u;
}

//

double myseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

//
//
//

void init(double*  p, int N, int M, int K, int K_loc, int k_start, int k_end)
{
//#pragma omp parallel for
	for (int k = k_start;  k <= k_end; ++k)
	{
		int global = k;
		int local  = k - k_start + 1;
		//printf("global = %d, local = %d, ks = %d, ke = %d, K = %d\n", global, local, k_start, k_end, K);
		if ((global == 1) || (global == K))
		{
			//printf("on the boundary k -> %d\n", k - k_start + 1);
			//if (k == 0) printf("*\n");
			//else if (k == K - 1) printf("**\n");
			for (int i = 0; i < N; i++)
			{
				for (int j = 0;  j < M; ++j)
				{
					p[local*M*N + i*M + j] = 1 + 0*global;
				}
			}
		}	
		else{ //printf("in the domain  -> %d\n", local); 
		for (int i = 0; i < N; i++)
		{
			for (int j = 0;  j < M; ++j)
			{
				if ( ((i == 0) || (j == 0)) || (k == 1) || (i == N - 1) || (j == M - 1) )
					p[local*M*N + i*M + j] = 1 + 0*global;
				else
                                        p[local*M*N + i*M + j] = 0.;
                        }
                }
		}
        }
}

//
//
//


double maxNorm(double* v1, double* v2, int size)
{

        double mymax = 0.;
#pragma omp parallel for reduction(max:mymax) 
        for (int ii = 0; ii < size; ++ii)
        {
		//printf("%d: %f %f\n", ii, v1[ii], v2[ii]); fflush(stdout);
                if (fabs(*v1 - *v2) > mymax)
                {
                        //printf("i = %d, v1 = %g, v2 = %g, max = %g, my_max = %g\n", ii, *v1, *v2, fabs(*v1 - *v2), mymax);
                        mymax = (double) fabs(*v1 - *v2);
                }
                ++v1; ++v2;
        }
        return mymax;
}

//
//
//

double l2Norm(double* v1, double* v2, int size)
{

        double myl2 = 0.;
#pragma omp parallel for reduction(+:myl2) 
        for (int ii = 0; ii < size; ++ii)
        {
		printf("%f %f\n", v1[ii], v2[ii]);
                myl2 += fabs(v1[ii]-v2[ii])*(v1[ii] - v2[ii]);
        }
        return sqrt(myl2)/size;
}

//
//
//

void print(double* p, int m, int n)
{
        for (int i = 0; i < MIN(n, 10); ++i)
        {
                for (int j = MIN(m, 10); j > 0; --j)
                {
                        printf("%e ", *p);
                        ++p;
                }
                p += m - MIN(m, 10);
                printf("\n");
        }
}
