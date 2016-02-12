#define MIN(a,b)  (((a<b)?a:b))

#ifdef __GNUC__
#include <immintrin.h>
#endif


#ifdef STREAMSTORE
#define SSE_STORE _mm_stream_pd
#define AVX_STORE _mm256_stream_pd
#else
#define SSE_STORE _mm_store_pd
#define AVX_STORE _mm256_store_pd
#endif


static int DIM_N   = 1001;
static int DIM_M   = 1001;

static double EPSI = 1.0e-3;

typedef double adouble __attribute__ ((aligned(16)));

double myseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


void init(double*  p, int n, int m)
{
        int i, j;
        for (j = 0; j < m; j++) 
        {
                for (i = 0; i < n; i++) 
                {
                        if (((i == 0) || (j == 0)) || (i == n - 1) || (j == m - 1))
                                p[j*n + i] = 1.;
                        else
                                p[j*n + i] = 0.;

                }
        }
}


double l2Norm(double* v1, double* v2, int size)
{
        double l2 = 0.;
        int ii;
        for (ii = 0; ii < size; ++ii, ++v1, ++v2)
        {
                //printf("Norm: i = %d, v1 = %g, v2 = %g, max = %g\n", ii, *v1, *v2, max);
                        l2 += (*v1 - *v2)*(*v1 - *v2);
        }
        return sqrt(l2);
}

double maxNorm(double* v1, double* v2, int size)
{

        double max = 0.;
        int ii;
        for (ii = 0; ii < size; ++ii, ++v1, ++v2)
        {
                //printf("Norm: i = %d, v1 = %g, v2 = %g, max = %g\n", ii, *v1, *v2, max);
                if (fabs(*v1 - *v2) > max)
                {
                        max = fabs(*v1 - *v2);
                }
        }
        return max;
}


__inline
void scheme(double* v1, double* v2, int n)
{
        double phi_e = *(v1 + 1);
        double phi_w = *(v1 - 1);

        double phi_n = *(v1 + n);
        double phi_s = *(v1 - n);

        double phi = 0.25*(phi_e + phi_w + phi_n + phi_s);

        *(v2) = phi;
}


__inline
void avx_scheme(adouble* v1, adouble* v2, int n)
{
        __m256d phi_e = _mm256_loadu_pd (v1 + 1 );
        __m256d phi_w = _mm256_loadu_pd (v1 - 1 );
        __m256d phi_n = _mm256_loadu_pd (v1 + n);
        __m256d phi_s = _mm256_loadu_pd (v1 - n);
        //                                        //
        __m256d alpha = _mm256_set1_pd(0.25);
        AVX_STORE(v2, alpha*(phi_e + phi_w + phi_n + phi_s));
        //
}


void print256(__m256d val)
{
        double a[4];
        _mm256_store_pd(&a[0], val);
        printf("%f %f %f %f", a[0], a[1], a[2], a[3]);

}

void print128(__m128d val)
{
        double a[2];
        _mm_store_pd(&a[0], val);
        printf("%f %f", a[0], a[1]);

}


void print(double* p, int n, int m)
{
        int i, j;
        double *pa, *pb = p;
        for (i=0; i < MIN(n, 11); ++i)
        {
                pa = pb;
                for (j=0; j < MIN(m, 11); ++j)
                {
                        printf("%e ", *pb);
                        ++pb;
                }
                pb = pa + m;
                printf("\n");
        }
}
