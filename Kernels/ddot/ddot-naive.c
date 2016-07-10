inline
double ddot(int N, double* a, int incx, double *b, int incy)
{
	int i;

	double c = 0.;
	for (i = 0; i < N/1; i++)
	{
		c += a[i + 0]*b[i + 0];
		//printf("%f %f: %f\n", a[i], b[i], c);
	}
	return c;
}
