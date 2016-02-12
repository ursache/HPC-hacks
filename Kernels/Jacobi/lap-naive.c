
inline
void kernel(double* v1, double * v2, int m)
{
	double phi_e = *(v1 + 1);
	double phi_w = *(v1 - 1);

	double phi_n = *(v1 + m);
	double phi_s = *(v1 - m);

	double phi = 0.25*(phi_e + phi_w + phi_n + phi_s);

	*(v2) = phi;
}


void laplacian(double* v1, double* v2, int dim_m, int dim_n)
{
	int m = dim_m;
	//
#pragma omp parallel for schedule(static)
	for (int j = 1; j < dim_m - 1; ++j )
	{
		for (int i = 1; i < dim_n - 1; ++i)
		{
#if 0
			kernel(v1 + j*dim_m + i, v2 + j*dim_m + i, dim_m);
#else 
			double phi_e = v1[(j + 0)*dim_m + i + 1];
			double phi_w = v1[(j + 0)*dim_m + i - 1];

			double phi_n = v1[(j + 1)*dim_m + i + 0];
			double phi_s = v1[(j - 1)*dim_m + i + 0];

			double phi = 0.25*(phi_e + phi_w + phi_n + phi_s);

			v2[j*dim_m + i] = phi;
#endif
		}
	}
}

