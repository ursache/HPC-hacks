#include "defs.h"

#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define MAX(a,b) (((a) >= (b)) ? (a) : (b)) 

extern "C" void upwind3(const  float uin[],
			const  float vin[],
			const  float win[],
			const  float factor,
			float uout[],
			float vout[],
			float wout[])
{
    for ( int k = 0; k < OUTPUTSTRIDE; k++)
    {
	const  int zin = k * INPUTSLICE;
	const  int zout = k * OUTPUTSLICE;
		
	for ( int j = 0; j < OUTPUTSTRIDE; j++)
	{
	    const  int basein = zin + j * INPUTSTRIDE;
	    const  int baseout = zout + j * OUTPUTSTRIDE;
			
	    for (int i = 0; i < OUTPUTSTRIDE; ++i)
	    {		
		const real uF2 = uin[basein + i - 2 * INPUTSLICE];
		const real uF  = uin[basein + i - INPUTSLICE];
		const real uS2 = uin[basein + i - 2 * INPUTSTRIDE];
		const real uS  = uin[basein + i - INPUTSTRIDE];
		const real uW2 = uin[basein + i - 2];
		const real uW  = uin[basein + i - 1];
		const real u = uin[basein + i];
		const real uE  = uin[basein + i + 1];
		const real uE2 = uin[basein + i + 2];
		const real uN  = uin[basein + i + INPUTSTRIDE];
		const real uN2 = uin[basein + i + 2 * INPUTSTRIDE];
		const real uB  = uin[basein + i + INPUTSLICE];
		const real uB2 = uin[basein + i + 2 * INPUTSLICE];
		const real u3 = 3.f * u;
		const real dudyS = 2.f * uN + u3 - 6.f * uS + uS2;
		const real dudxW = 2.f * uE + u3 - 6.f * uW + uW2;
		const real dudxE = -uE2 + 6.f * uE - u3 - 2.f * uW;
		const real dudyN = -uN2 + 6.f * uN - u3 - 2.f * uS;
		const real dudzF = 2.f * uB + u3 - 6.f * uF + uF2;
		const real dudzB = -uB2 + 6.f * uB - u3 - 2.f * uF;
		const real minu = MIN(u, (real)0);
		const real maxu = MAX(u, (real)0);

		const real vF2 = vin[basein + i - 2 * INPUTSLICE];
		const real vF  = vin[basein + i - INPUTSLICE];
		const real vS2 = vin[basein + i - 2 * INPUTSTRIDE];
		const real vS  = vin[basein + i - INPUTSTRIDE];
		const real vW2 = vin[basein + i - 2];
		const real vW  = vin[basein + i - 1];
		const real v = vin[basein + i];
		const real vE  = vin[basein + i + 1];
		const real vE2 = vin[basein + i + 2];
		const real vN  = vin[basein + i + INPUTSTRIDE];
		const real vN2 = vin[basein + i + 2 * INPUTSTRIDE];
		const real vB  = vin[basein + i + INPUTSLICE];
		const real vB2 = vin[basein + i + 2 * INPUTSLICE];
		const real v3 = 3.f * v;
		const real dvdyS = 2.f * vN + v3 - 6.f * vS + vS2;
		const real dvdxW = 2.f * vE + v3 - 6.f * vW + vW2;
		const real dvdxE = -vE2 + 6.f * vE - v3 - 2.f * vW;
		const real dvdyN = -vN2 + 6.f * vN - v3 - 2.f * vS;
		const real dvdzF = 2.f * vB + v3 - 6.f * vF + vF2;
		const real dvdzB = -vB2 + 6.f * vB - v3 - 2.f * vF;
		const real minv = MIN(v, (real)0);
		const real maxv = MAX(v, (real)0);

		const real wF2 = win[basein + i - 2 * INPUTSLICE];
		const real wF  = win[basein + i - INPUTSLICE];
		const real wS2 = win[basein + i - 2 * INPUTSTRIDE];
		const real wS  = win[basein + i - INPUTSTRIDE];
		const real wW2 = win[basein + i - 2];
		const real wW  = win[basein + i - 1];
		const real w = win[basein + i];
		const real wE  = win[basein + i + 1];
		const real wE2 = win[basein + i + 2];
		const real wN  = win[basein + i + INPUTSTRIDE];
		const real wN2 = win[basein + i + 2 * INPUTSTRIDE];
		const real wB  = win[basein + i + INPUTSLICE];
		const real wB2 = win[basein + i + 2 * INPUTSLICE];
		const real w3 = 3.f * w;
		const real dwdyS = 2.f * wN + w3 - 6.f * wS + wS2;
		const real dwdxW = 2.f * wE + w3 - 6.f * wW + wW2;
		const real dwdxE = -wE2 + 6.f * wE - w3 - 2.f * wW;
		const real dwdyN = -wN2 + 6.f * wN - w3 - 2.f * wS;
		const real dwdzF = 2.f * wB + w3 - 6.f * wF + wF2;
		const real dwdzB = -wB2 + 6.f * wB - w3 - 2.f * wF;
		const real minw = MIN(w, (real)0);
		const real maxw = MAX(w, (real)0);

		const real uresult = factor * (
		    maxu * dudxW + minu * dudxE +
		    maxv * dudyS + minv * dudyN +
		    maxw * dudzF + minw * dudzB);

		uout[baseout + i] = uresult;
				
		const real vresult = factor * (
		    maxu * dvdxW + minu * dvdxE +
		    maxv * dvdyS + minv * dvdyN +
		    maxw * dvdzF + minw * dvdzB );

		vout[baseout + i] = vresult;

		const real wresult = factor * (
		    maxu * dwdxW + minu * dwdxE +
		    maxv * dwdyS + minv * dwdyN +
		    maxw * dwdzF + minw * dwdzB );
		
		wout[baseout + i] = wresult;
	    }
	}
    }
}
