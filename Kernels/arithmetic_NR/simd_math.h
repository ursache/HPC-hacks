#ifndef SIMD_MATH
#define SIMD_MATH_
//
#include <immintrin.h>
//
inline __m256d RCP(const __m256d d)
{
        const __m128 b    = _mm256_cvtpd_ps(d);
        const __m128 rcp = _mm_rcp_ps (b);
        __m256d x0       = _mm256_cvtps_pd(rcp);
        //
        return x0;
}
//
//
//
inline __m256d RCP_1NR(const __m256d d)
{
        const __m128 b    = _mm256_cvtpd_ps(d);
        const __m128 rcp = _mm_rcp_ps (b);
        __m256d x0       = _mm256_cvtps_pd(rcp);
        //
        x0 = x0 + x0 - d*x0*x0;
        //                        //
        return x0;
}
//
//
//
inline __m256d RCP_2NR(const __m256d d)
{
        const __m128 b    = _mm256_cvtpd_ps(d);
        const __m128 rcp = _mm_rcp_ps (b);
        __m256d x0       = _mm256_cvtps_pd(rcp);
        //
        x0 = x0 + x0 - d*x0*x0;
        x0 = x0 + x0 - d*x0*x0;
        //
        return x0;
}
#endif
