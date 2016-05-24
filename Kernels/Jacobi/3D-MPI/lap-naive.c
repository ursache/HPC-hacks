// 
// * author gilles fourestey gilles.fourestey@epfl.ch
// * Copyright 2015. All rights reserved.
// *
// * Users are NOT authorized
// * to employ the present software for their own publications
// * before getting a written permission from the author of this file
//
inline
void kernel(double* v1, double * v2, int m, int n)
{
	double phi_e = *(v1 + 1);
	double phi_w = *(v1 - 1);

	double phi_n = *(v1 + m);
	double phi_s = *(v1 - m);

	double phi_u = *(v1 + m*n);
	double phi_d = *(v1 - m*n);

	double phi = 1./6.*(phi_e + phi_w + phi_n + phi_s + phi_u + phi_d);

	*(v2) = phi;
}


void laplacian(double* v1, double* v2, int dim_m, int dim_n, int dim_k)
{
	int m = dim_m;
	//
#pragma omp parallel for schedule(static)
	for (int k = 1; k < dim_k + 1; ++k)
	{
		for (int j = 1; j < dim_m - 1; ++j)
		{
			for (int i = 1; i < dim_n - 1; ++i)
			{
				kernel(v1 + k*dim_n*dim_m + j*dim_m + i, v2 + k*dim_n*dim_m + j*dim_m + i, dim_m, dim_n);
			}
		}
	}
}

