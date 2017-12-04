#include<stdlib.h>
#include<time.h>
#include<stdio.h>
#include<sys/time.h>
#include<immintrin.h>
#include<mm_malloc.h>

#include "utils.h"

#define NN     100000
#define NTIMES 100 
//
//
//
struct AOS 
{ 	
	double x; 	
	double y; 
	double z;
}; 
//
// Cross Product pure function
static inline
struct AOS CrossProduct_AOS_pure(struct AOS* a, struct AOS* b)
{
	// 
	struct AOS c;
	//
	c.x = a->y*b->z - a->z*b->y;
	c.y = a->z*b->x - a->x*b->z;
	c.z = a->x*b->y - a->y*b->x;
	//printf("%f %f %f ^ %f %f %f = %f %f %f\n", a->x, a->y, a->z, b->x, b->y, b->z, c.x, c.y, c.z);
	//
	return c;
}
//
void CrossProduct_AOS(struct AOS* c, struct AOS* a, struct AOS* b, int size)
{
	int ii = 0;
	for (; ii < size; ++ii)
	{
		c[ii] = CrossProduct_AOS_pure(&a[ii], &b[ii]);
		//printf("%f %f %f\n", c[ii].x, c[ii].y, c[ii].z);
	}
}
//
// Cross Product pure AVX function
static inline 
__m256d CrossProduct_AOS_AVX_pure(__m256d a, __m256d b)
{
	__m256d ap1 = _mm256_permute4x64_pd(a, 0b00001001);
	__m256d bp1 = _mm256_permute4x64_pd(b, 0b00010010);
	
	__m256d ap2 = _mm256_permute4x64_pd(a, 0b00010010);
	__m256d bp2 = _mm256_permute4x64_pd(b, 0b00001001);
	
	return _mm256_sub_pd(_mm256_mul_pd(ap1, bp1), _mm256_mul_pd(ap2, bp2));
 }

void 
CrossProduct_AOS_AVX(struct AOS* c, struct AOS* a, struct AOS* b, int size)
{
        int ii = 0;
        for (; ii < size - size%3; ii = ii + 3)
        {
		__m256d _a = _mm256_loadu_pd((void*) &a[ii]);
		__m256d _b = _mm256_loadu_pd((void*) &b[ii]);
		//
		_mm256_storeu_pd((void*) &c[ii], CrossProduct_AOS_AVX_pure(_a, _b));
        }
	for (; ii < size; ++ii)
	{
		c[ii] = CrossProduct_AOS_pure(&a[ii], &b[ii]);
	}
}
//
//
//
void
fill_AOS(struct AOS* aos, int size)
{
	//
	time_t t;
	int ii = 0;
	srand(100*time(&t));
	for(; ii < size; ++ii)
	{
		aos[ii].x = randomfloat(); 
		aos[ii].y = randomfloat(); 
		aos[ii].z = randomfloat(); 
		//printf("%f %f %f\n", aos[ii].x, aos[ii].y, aos[ii].z);
	}
}
//
// SOA
//
struct SOA
{
	double* x;
	double* y;
	double* z;
};
// AOS to SOA
void
AOS_to_SOA(struct SOA* soa, struct AOS* aos, int size)
{
        int ii = 0;
        for (; ii < size; ++ii)
        {
                soa->x[ii] = aos[ii].x;
                soa->y[ii] = aos[ii].y;
                soa->z[ii] = aos[ii].z;
		//printf("%f %f %f\n", soa->x[ii], soa->y[ii], soa->z[ii]);
        }
}
//
//
//
static inline
void
CrossProduct_SOA_pure(double* cx, double* cy, double* cz, double *ax, double* ay, double* az , double *bx, double* by, double* bz)
{
	//
	//
	*cx = (*ay)*(*bz) - (*az)*(*by);
	*cy = (*az)*(*bx) - (*ax)*(*bz);
	*cz = (*ax)*(*by) - (*ay)*(*bx);
}

void CrossProduct_SOA(struct SOA* c, struct SOA* a, struct SOA* b, int size)
{
        int ii = 0;
        for (; ii < size; ++ii)
        {
                CrossProduct_SOA_pure(&c->x[ii], &c->y[ii], &c->z[ii], &a->x[ii], &a->y[ii], &a->z[ii], &b->x[ii], &b->y[ii], &b->z[ii]);
        }
}
//
//
//
static inline
void
CrossProduct_SOA_AVX_pure(__m256d* cx, __m256d* cy, __m256d* cz, __m256d* ax, __m256d* ay, __m256d* az, __m256d* bx, __m256d* by, __m256d* bz)
{
        *cx = _mm256_sub_pd(_mm256_mul_pd(*ay, *bz), _mm256_mul_pd(*az, *by));
        *cy = _mm256_sub_pd(_mm256_mul_pd(*az, *bx), _mm256_mul_pd(*ax, *bz));
        *cz = _mm256_sub_pd(_mm256_mul_pd(*ax, *by), _mm256_mul_pd(*ay, *bx));
}
//
void
CrossProduct_SOA_AVX(struct SOA* c, struct SOA* a, struct SOA* b, int size)
{
        int ii = 0;
        for (; ii < size - size%4; ii = ii + 4)
        {
		__m256d _cx, _cy, _cz;
		//
                __m256d _ax = _mm256_loadu_pd((void*) &a->x[ii]);
                __m256d _ay = _mm256_loadu_pd((void*) &a->y[ii]);
                __m256d _az = _mm256_loadu_pd((void*) &a->z[ii]);
                __m256d _bx = _mm256_loadu_pd((void*) &b->x[ii]);
                __m256d _by = _mm256_loadu_pd((void*) &b->y[ii]);
                __m256d _bz = _mm256_loadu_pd((void*) &b->z[ii]);
                //
		CrossProduct_SOA_AVX_pure(&_cx, &_cy, &_cz, &_ax, &_ay, &_az, &_bx, &_by, &_bz);
		//
                _mm256_storeu_pd((void*) &c->x[ii], _cx); 
        }
        for (; ii < size; ++ii)
        {
                CrossProduct_SOA_pure(&c->x[ii], &c->y[ii], &c->z[ii], &a->x[ii], &a->y[ii], &a->z[ii], &b->x[ii], &b->y[ii], &b->z[ii]);
        }
}
//
//
//
int main()
{
	printf("Cross product, array size = %f GB\n", NN*8/1024./1024./1024.); 
	//	
	struct AOS a0 = {1., 0., 0.};
	struct AOS b0 = {.x = 0., .y = 1., .z = 0.};
	struct AOS c0;
	//
	c0 = CrossProduct_AOS_pure(&a0, &b0);
	printf("AOS Sanity check: {1., 0., 0.}^{0., 1., 0.} = {%f, %f, %f} (sould be {0, 0, 1})\n", c0.x, c0.y, c0.z);
	//
	struct AOS* a_aos;
	struct AOS* b_aos;
	struct AOS* c_aos;
	a_aos = (struct AOS*) _mm_malloc(NN*sizeof(struct AOS), 32);
	b_aos = (struct AOS*) _mm_malloc(NN*sizeof(struct AOS), 32);
	c_aos = (struct AOS*) _mm_malloc(NN*sizeof(struct AOS), 32);
	//	
	fill_AOS(a_aos, NN);
	fill_AOS(b_aos, NN);
	//fill_AOS(c_aos, NN);
	//
	double time;
	unsigned long long    tic, toc;
	//
	printf("AOS: \n");
	time = -myseconds(); 
	tic = rdtsc();	
	for (int ii = 0; ii < NTIMES; ++ii)
		CrossProduct_AOS    (c_aos, a_aos, b_aos, (int) NN);	
	toc = rdtsc();
	time += myseconds(); 
	printf("cross product AOS    : %lf GB/s, %f flops/cycles\n", NTIMES*NN*8./time/1024./1024./1024., 9.*NTIMES*NN/(toc - tic));	
	//
	time = -myseconds();
	tic = rdtsc();	
	for (int ii = 0; ii < NTIMES; ++ii)
		CrossProduct_AOS_AVX(c_aos, a_aos, b_aos, (int) NN);
	toc = rdtsc();
	time += myseconds();
	printf("cross product AOS AVX: %lf GB/s, %f flops/cycles\n", NTIMES*NN*8./time/1024./1024./1024., 9.*NTIMES*NN/(toc - tic));	
	//
	// SOA
	//
	struct SOA a_soa, b_soa, c_soa;
	a_soa.x = (double*) _mm_malloc(NN*sizeof(double), 32);
	a_soa.y = (double*) _mm_malloc(NN*sizeof(double), 32);
	a_soa.z = (double*) _mm_malloc(NN*sizeof(double), 32);
	//
	b_soa.x = (double*) _mm_malloc(NN*sizeof(double), 32);
	b_soa.y = (double*) _mm_malloc(NN*sizeof(double), 32);
	b_soa.z = (double*) _mm_malloc(NN*sizeof(double), 32);
	//
	c_soa.x = (double*) _mm_malloc(NN*sizeof(double), 32);
        c_soa.y = (double*) _mm_malloc(NN*sizeof(double), 32);
        c_soa.z = (double*) _mm_malloc(NN*sizeof(double), 32);
	//
	AOS_to_SOA(&a_soa, a_aos, NN);
	AOS_to_SOA(&b_soa, b_aos, NN);
	AOS_to_SOA(&c_soa, c_aos, NN);
	//
	printf("SOA: \n");
	time = -myseconds();
	tic = rdtsc();
	for (int ii = 0; ii < NTIMES; ++ii)
        	CrossProduct_SOA(&c_soa, &a_soa, &b_soa, NN);
	toc = rdtsc();
        time += myseconds();
	printf("cross product SOA    : %lf GB/s, %f flops/cycle\n", NTIMES*NN*8./time/1024./1024./1024., 9.*NTIMES*NN/(toc - tic)); 
	//
        time = -myseconds();
        tic = rdtsc();
	for (int ii = 0; ii < NTIMES; ++ii)
		CrossProduct_SOA_AVX(&c_soa, &a_soa, &b_soa, NN);
        toc = rdtsc();
        time += myseconds();
        printf("cross product SOA AVX: %lf GB/s, %f flops/cycle\n", NTIMES*NN*8./time/1024./1024./1024., 9.*NTIMES*NN/(toc - tic));
	//
	fflush(stdout);
	_mm_free(a_aos);
	_mm_free(b_aos);
	_mm_free(c_aos);

	_mm_free(a_soa.x);
	_mm_free(a_soa.y);
	_mm_free(a_soa.z);

	_mm_free(b_soa.x);
	_mm_free(b_soa.y);
	_mm_free(b_soa.z);

	_mm_free(c_soa.x);
	_mm_free(c_soa.y);
	_mm_free(c_soa.z);
};


