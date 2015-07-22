#include <math.h>
#include <time.h>

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
  	double mp = 1e-4;
	printf("norm = %g\n", mp);
	int i;
	int j;

	double norm = 0.;

//#pragma omp parallel for private(i, j, norm)
    for (j = 0; j < N; j++)
    	{
        for (i = 0; i < M; i++)
          {
            //printf("%g = %g %g %g\n", mat[i+j*M], mat_ref[i + j*M], fabs(mat[i + j*M ] - mat_ref[i + j*M ]), norm) ;
            if (fabs(mat[i + j*M ] - mat_ref[i + j*M ]) > mp)
              {
		      //printf("line %d, col %d = %g, should be %g\n", i, j, mat[i + j*M], mat_ref[i + j*M]); 
		      norm = fabs((double)mat[i + j*M] - (double)mat_ref[i + j*M]);
				//if ((i==0) && (j==0))		printf("line %d, col %d = %g, should be %g\n", i, j, mat[i + j*M], mat_ref[i + j*M]); 
		      if (norm != 0.)
			      printf("line %d, col %d = %g, should be %g\n", i, j, mat[i + j*M], mat_ref[i + j*M]); 
					//exit(-1);
              }
          }
//        std::cout << "----" << std::endl;
      }
    return norm;
}

static void fill0(double* A, int size1, int size2)
{
    int ii, jj;
#pragma omp parallel for private(ii, jj)
    for (ii = 0; ii < size1; ++ii)
		for (jj = 0; jj < size2; ++jj)
		{
			A[ii*size1 + jj] = 1;
		}
}

static void fill(double *A, int size, double v)
{
	int ii;
//#pragma omp parallel for private(ii)
  	for (ii = 0; ii < size; ++ii)
	{
		double var =  v*rand()/(RAND_MAX+1.); 
		//printf("var = %f\n", var);
		A[ii] = var;
	}
}

static void zfill(double2 *A, int size, double2 v)
{
        int ii;
        for (ii = 0; ii < size; ++ii)
        A[ii] = v;
}

static void eye(double *A, int lda, int n)
{
   	fill(A, lda*n, 0.);
	int ii;

   	for (ii = 0; ii < lda; ++ii)
    	A[ii*lda + ii] = 1.;
}

static void zeye(double2 *A, int lda, int n)
{
	double2 zero; zero.x = 0.; zero.y = 0.;
        zfill(A, lda*n, zero);
        int ii;

        for (ii = 0; ii < lda; ++ii)
        A[ii*lda + ii].x = 1.;
}

