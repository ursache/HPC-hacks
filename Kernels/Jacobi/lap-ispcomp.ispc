export void laplacian_orig(  uniform double v1[], 
		 	uniform double v2[], 
			uniform int dim_m,
			uniform int dim_n)
{
	//int i, j;

	//foreach(i = 1 ... dim_m - 1, j = 1 ... dim_n - 1)
	//for (j = 1; j < dim_m - 1; ++j)
	cfor(int j = 1; j < dim_n - 1; ++j)
		//for (i = 1; i < dim_n - 1; ++i)
		foreach (i = 1 ... dim_n - 1)
		{
			double* up = v1 + j*dim_m + i;
			//
			double phi_w = *(up - 1);
			double phi_e = *(up + 1);
			double phi_n = *(up + dim_m );
			double phi_s = *(up - dim_m ) ;
			//
			*(v2 + j*dim_m + i) = 0.25*(phi_w + phi_n + phi_e + phi_s);
		}
}


export void laplacian_v1( const uniform double v1[],
                       uniform double v2[],
                       uniform int dim_m,
                       uniform int dim_n)
{
    cfor(int y = 1; y < dim_n - 1; ++y)
    {
        int base = y * dim_m;
        int basem1 = (y - 1) * dim_m;
        int basep1 = (y + 1) * dim_m;

        foreach(x = 1 ... dim_n - 1)
        {
            const double w = v1[base + x - 1];
            const double e = v1[base + x + 1];
            const double n = v1[basep1 + x];
            const double s = v1[basem1 + x];

            v2[base + x] = 0.25 * (w + e + n + s);
        }
    }
}


export void laplacian_v2( const uniform double v1[],
                       uniform double v2[],
                       uniform int dim_m,
                       uniform int dim_n)
{
    cfor(int y = 1; y < dim_n - 1; ++y)
    {
        int base = y * dim_m;
        int basem1 = (y - 1) * dim_m;
        int basep1 = (y + 1) * dim_m;

        foreach(x = 0 ... dim_n)
        {
            cif(x > 0 && x < dim_n - 1)
            {
                const double w = v1[base + x - 1];
                const double e = v1[base + x + 1];
                const double n = v1[basep1 + x];
                const double s = v1[basem1 + x];

                v2[base + x] = 0.25 * (w + e + n + s);
            }
        }
    }
}


export void laplacian_v3( const uniform double v1[],
                       uniform double v2[],
                       uniform int dim_m,
                       uniform int dim_n)
{
    cfor(uniform int y = 1; y < dim_n - 1; ++y)
    {
        const uniform int base = y * dim_m;
        const uniform int basem1 = (y - 1) * dim_m;
        const uniform int basep1 = (y + 1) * dim_m;

        foreach(x = 1 ... dim_n - 1)
        {
                const double w = v1[base + x - 1];
                const double e = v1[base + x + 1];
                const double n = v1[basep1 + x];
                const double s = v1[basem1 + x];

                v2[base + x] = 0.25 * (w + e + n + s);
        }
    }
}



export void laplacian_v4( const uniform double v1[],
                       uniform double v2[],
                       uniform int dim_m,
                       uniform int dim_n)
{
    cfor(uniform int y = 2; y < dim_n - 1; ++y)
    {
        const uniform int ybase = y * dim_m;
        const uniform int ybasem1 = (y - 1) * dim_m;
        const uniform int ybasep1 = (y + 1) * dim_m;
	//
        cfor(uniform int xbase = 0; xbase < dim_n; xbase += programCount)
        {
            const int x = xbase + programIndex;

            cif(x > 0 && x < dim_n - 1)
            {
                const double w = v1[ybase + x - 1];
                const double e = v1[ybase + x + 1];
                const double n = v1[ybasep1 + x];
                const double s = v1[ybasem1 + x];

                v2[ybase + x] = 0.25 * (w + e + n + s);
            }
        }
    }
}



export void laplacian_v5( const uniform double v1[], uniform double v2[],
                   const uniform int dim_m, const uniform int dim_n)
{
    uniform const int stride = dim_m + 2;

    cfor(uniform int y = 0; y < dim_n; ++y)
    {
        const uniform int base = y * stride;
        foreach(x = 0 ... dim_n)
        {
            const double s = v1[base + x - stride];
            const double w = v1[base + x - 1];
            const double e = v1[base + x + 1];
            const double n = v1[base + x + stride];

            v2[base + x] = 0.25 * (w + e + n + s);
        }
    }
}


export void laplacian_v6( const uniform double v1[], uniform double v2[],
		const uniform int dim_m, const uniform int dim_n, const uniform int ldim)
{
    uniform const int stride = ldim;

    cfor(uniform int y = 0; y < dim_n; ++y)
    {
        const uniform int base = y * stride;

        foreach(x = 0 ... dim_m)
        {
            const double s = v1[base + x - stride];
            const double w = v1[base + x - 1];
            const double e = v1[base + x + 1];
            const double n = v1[base + x + stride];

            v2[base + x] = 0.25 * (w + e + n + s);
        }
    }
}

