//author gilles.fourestey@epfl.ch
void dgemm( const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double* C, const int ldc)
{
	int i, j, k;

	for (j = 0; j < N; ++j)
		for (i = 0; i < M; ++i)
		{
			double cij = 0.; 
			for (k = 0; k < K; ++k) 
				cij += A[i + k*lda]*B[k + j*ldb];
			C[j*ldc + i] += cij;	
		}
}
