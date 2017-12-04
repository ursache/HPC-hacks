#include <immintrin.h>
#include <stdio.h>

int
main()
{
	__m256d a = {1, 2, 3, 4};
	{
		__m256d b = _mm256_permute4x64_pd(a, 0b00001001); 
		double* bb = (double*) &b;
		printf("%f %f %f %f\n", bb[0], bb[1], bb[2], bb[3]);
	}
	{
		__m256d b = _mm256_permute4x64_pd(a, 0b00010010); 
		//__m256d b = _mm256_shuffle_pd(a, a, _MM_SHUFFLE(4, 1, 2, 3));
		double* bb = (double*) &b;
		printf("%f %f %f %f\n", bb[0], bb[1], bb[2], bb[3]);
	}
}
