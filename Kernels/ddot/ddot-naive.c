double ddot(int N, double *a, int incx, double *b, int incy)
{
	int i;

	double c = 0.;
#pragma omp parallel for private(i) reduction(+:c)
	for (i = 0; i < N/1; i++)
	{
		c += a [i + 0]*b [i + 0];
	}
	return c;
}
