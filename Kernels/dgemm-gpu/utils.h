#include <math.h>

static void simple_dgemm(int n, double alpha, const double *A, const double *B,
                         double beta, double *C)
{
    int i;
    int j;
    int k;
    for (i = 0; i < n; ++i)
      {
        for (j = 0; j < n; ++j)
          {
            double prod = 0;
            for (k = 0; k < n; ++k)
              {
                prod += A[k * n + i] * B[j * n + k];
              }
            //std::cout << prod << std::endl;
            C[j * n + i] = alpha*prod + beta*C[j*n + i];
          }
      }
}



double verifyResult(const double *mat, const double *mat_ref, int M, int N)
{
  	double norm;
	int i;
	int j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			//printf("%d %d: %f %f\n", i, j, mat[i+j*M], mat_ref[i + j*M]);
			norm = fabs((double)mat[i + j*M] - (double)mat_ref[i + j*M]);
			if (norm > 1.e-5)
			{
				//if (norm > 1.e-6)
					printf("%d %d = %f, should be %f\n", i, j, mat[i + j*M], mat_ref[i + j*M]); 
			}
		}
		//        std::cout << "----" << std::endl;
	}
	return norm;
}


void fill(double *A, int size, double v)
{
	if (v == 0.)
	{
		memset(A, size*sizeof(double), 0.);
		return;
	}
	long int ii;
	for (ii = 0; ii < size; ++ii)
	{
		double r1 = (double)(1.+rand())/(double)(RAND_MAX+1.);
		double r2 = (double)(1.+rand())/(double)(RAND_MAX+1.);
		A[ii] = 0. + 1*(double)(sqrt(-10.0*log(r1))*cos(2.0*M_PI*r2));
	}
}

static void eye(double *A, int lda, int n)
{
	fill(A, lda*n, 0.);
	int ii;

	for (ii = 0; ii < lda; ++ii)
		A[ii*lda + ii] = 1.;
}
