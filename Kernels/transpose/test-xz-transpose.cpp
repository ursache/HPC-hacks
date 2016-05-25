/*
* test-xz-transpose.cpp 
*
* Created and authored by Diego Rossinelli on 2015-11-25.
* Copyright 2015. All rights reserved.
*
* Users are NOT authorized
* to employ the present software for their own publications
* before getting a written permission from the author of this file.
*/


#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <limits>

//#include "../aiw3-gen2.h"

using namespace std;

//this is used to reject or accept the tests below
bool compare_vectors(vector<float>& data, vector<float>& refdata, double tolfactor)
{
    printf("comparing sizes: %d %d\n", data.size(), refdata.size());
    if (data.size() != refdata.size())
	return false;

    double maxerr = 0;
    const double tol = numeric_limits<float>::epsilon() * tolfactor;
    
    for(int i = 0; i < data.size(); ++i)
    {
	double a = data[i];
	double b = refdata[i];
	double err = a - b;
	double relerr = err / max(max(fabs(a), fabs(b)), 1e-15);
	maxerr = fmax(fabs(err), maxerr);
	if (err > tol && relerr > tol)
	{
	    printf("Significant difference at entry %d: val: %e, ref: %e, err: %e, rel err: %e\n", i, a, b, err, relerr);

	    
	    return false;
	}
    }

    printf("maxerr: %e\n", maxerr);
    return true;
}
/*
void test_3d(int nx, int ny, int nz)
{
    const int n = nx * ny * nz;
    vector<float> data(n);

    for(int i = 0; i < n; ++i)
	data[i] = drand48();

    vector<float> ref = data;
    
    int xc, yc, zc;
    aiw3_gen2_fw(&data.front(), nx, ny, nz, xc, yc, zc);

    printf("3d fwt done: %d %d %d\n", xc, yc, zc);

    aiw3_gen2_bw(&data.front(), nx, ny, nz, xc, yc, zc);

    bool success = compare_vectors(data, ref, 9e2);
    assert(success);
}
*/
extern "C" void xz_transpose_8x8x8 (float *data);

void test_transpose()
{
    printf("testing transposition\n");

    float cube[8][8][8];

    for(int iz = 0; iz < 8; ++iz)
	for(int iy = 0; iy < 8; ++iy)
	    for(int ix = 0; ix < 8; ++ix)
		cube[iz][iy][ix] = ix + 8 * (iy + 8 * iz);

    xz_transpose_8x8x8((float *)cube);

    bool fail = false;
    for(int iz = 0; iz < 8; ++iz)
	for(int iy = 0; iy < 8; ++iy)
	    for(int ix = 0; ix < 8; ++ix)
		if (cube[ix][iy][iz] != ix + 8 * (iy + 8 * iz))
		{
		    printf("ooops %d %d %d should have been %f, instead is %d\n",
			   ix, iy, iz, cube[ix][iy][iz], ix + 8 * (iy + 8 * iz));

		    fail |= true;
		}

    if (fail)
	abort();
    
    printf("all good\n");
}

extern "C" void xz_transpose(
    const float * const src,
    const int xsize,
    const int ysize,
    const int zsize,
    float * const dst);

void test_transpose2()
{
    printf("testing transposition more difficult...\n");

    const int ny = 41;
    const int nz = 92;
    const int nx = 155;
    
    float * data = new float[nx * ny * nz];

    for(int iz = 0; iz < nz; ++iz)
	for(int iy = 0; iy < ny; ++iy)
	    for(int ix = 0; ix < nx; ++ix)
		data[ix + nx * (iy + ny * iz)] = ix + nx * (iy + ny * iz);

    float * tmp = new float[nx * ny * nz];

    xz_transpose(data, nx, ny, nz, tmp);
    
    bool fail = false;
    for(int iz = 0; iz < nz; ++iz)
	for(int iy = 0; iy < ny; ++iy)
	    for(int ix = 0; ix < nx; ++ix)
		if (tmp[iz + nz * (iy + ny * ix)] != data[ix + nx * (iy + ny * iz)])
		{
		    printf("ooops %d %d %d should have been %f, instead is %f\n",
			   ix, iy, iz, tmp[iz + nz * (iy + ny * ix)], data[ix + nx * (iy + ny * iz)]);

		    fail |= true;
		}

     xz_transpose(tmp, nz, ny, nx, data);
     
     for(int iz = 0; iz < nz; ++iz)
	for(int iy = 0; iy < ny; ++iy)
	    for(int ix = 0; ix < nx; ++ix)
		if (ix + nx * (iy + ny * iz) != data[ix + nx * (iy + ny * iz)])
		{
		    printf("ooops %d %d %d should have been %f, instead is %f\n",
			   ix, iy, iz, ix + nx * (iy + ny * iz), data[ix + nx * (iy + ny * iz)]);

		    fail |= true;
		}
     
     
     delete [] tmp;
     delete [] data;
     
    if (fail)
	abort();
    
    printf("all good\n");
}

int main()
{
    srand48(15);

    const int n = 52; //random number
    //  test_3d(2 * n, n, n * 3);

    test_transpose();
    test_transpose2();
    return 0;
}
