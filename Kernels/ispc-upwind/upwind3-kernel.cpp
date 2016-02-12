//author: diego rossinelli

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
		const float uF2 = uin[basein + i - 2 * INPUTSLICE];
		const float vF2 = vin[basein + i - 2 * INPUTSLICE];
		const float wF2 = win[basein + i - 2 * INPUTSLICE];

		const float uF  = uin[basein + i - INPUTSLICE];
		const float vF  = vin[basein + i - INPUTSLICE];
		const float wF  = win[basein + i - INPUTSLICE];

		const float uS2 = uin[basein + i - 2 * INPUTSTRIDE];
		const float vS2 = vin[basein + i - 2 * INPUTSTRIDE];
		const float wS2 = win[basein + i - 2 * INPUTSTRIDE];

		const float uS  = uin[basein + i - INPUTSTRIDE];
		const float vS  = vin[basein + i - INPUTSTRIDE];
		const float wS  = win[basein + i - INPUTSTRIDE];

		const float uW2 = uin[basein + i - 2];
		const float vW2 = vin[basein + i - 2];
		const float wW2 = win[basein + i - 2];

		const float uW  = uin[basein + i - 1];
		const float vW  = vin[basein + i - 1];
		const float wW  = win[basein + i - 1];

		const float u = uin[basein + i];
		const float v = vin[basein + i];
		const float w = win[basein + i];
		
		const float uE  = uin[basein + i + 1];
		const float vE  = vin[basein + i + 1];
		const float wE  = win[basein + i + 1];

		const float uE2 = uin[basein + i + 2];
		const float vE2 = vin[basein + i + 2];
		const float wE2 = win[basein + i + 2];

		const float uN  = uin[basein + i + INPUTSTRIDE];
		const float vN  = vin[basein + i + INPUTSTRIDE];
		const float wN  = win[basein + i + INPUTSTRIDE];

		const float uN2 = uin[basein + i + 2 * INPUTSTRIDE];
		const float vN2 = vin[basein + i + 2 * INPUTSTRIDE];
		const float wN2 = win[basein + i + 2 * INPUTSTRIDE];

		const float uB  = uin[basein + i + INPUTSLICE];
		const float vB  = vin[basein + i + INPUTSLICE];
		const float wB  = win[basein + i + INPUTSLICE];

		const float uB2 = uin[basein + i + 2 * INPUTSLICE];
		const float vB2 = vin[basein + i + 2 * INPUTSLICE];
		const float wB2 = win[basein + i + 2 * INPUTSLICE];

		const float u3 = 3.f * u;
		const float v3 = 3.f * v;
		const float w3 = 3.f * w;

		const float dudyS = 2.f * uN + u3 - 6.f * uS + uS2;
		const float dvdyS = 2.f * vN + v3 - 6.f * vS + vS2;
		const float dwdyS = 2.f * wN + w3 - 6.f * wS + wS2;

		const float dudxW = 2.f * uE + u3 - 6.f * uW + uW2;
		const float dvdxW = 2.f * vE + v3 - 6.f * vW + vW2;
		const float dwdxW = 2.f * wE + w3 - 6.f * wW + wW2;

		const float dudxE = -uE2 + 6.f * uE - u3 - 2.f * uW;
		const float dvdxE = -vE2 + 6.f * vE - v3 - 2.f * vW;
		const float dwdxE = -wE2 + 6.f * wE - w3 - 2.f * wW;

		const float dudyN = -uN2 + 6.f * uN - u3 - 2.f * uS;
		const float dvdyN = -vN2 + 6.f * vN - v3 - 2.f * vS;
		const float dwdyN = -wN2 + 6.f * wN - w3 - 2.f * wS;

		const float dudzF = 2.f * uB + u3 - 6.f * uF + uF2;
		const float dvdzF = 2.f * vB + v3 - 6.f * vF + vF2;
		const float dwdzF = 2.f * wB + w3 - 6.f * wF + wF2;

		const float dudzB = -uB2 + 6.f * uB - u3 - 2.f * uF;
		const float dvdzB = -vB2 + 6.f * vB - v3 - 2.f * vF;
		const float dwdzB = -wB2 + 6.f * wB - w3 - 2.f * wF;

		const float minu = MIN(u, 0.f);
		const float maxu = MAX(u, 0.f);
		const float minv = MIN(v, 0.f);
		const float maxv = MAX(v, 0.f);
		const float minw = MIN(w, 0.f);
		const float maxw = MAX(w, 0.f);

		const float uresult = factor * (
		    maxu * dudxW + minu * dudxE +
		    maxv * dudyS + minv * dudyN +
		    maxw * dudzF + minw * dudzB);
				
		const float vresult = factor * (
		    maxu * dvdxW + minu * dvdxE +
		    maxv * dvdyS + minv * dvdyN +
		    maxw * dvdzF + minw * dvdzB );

		const float wresult = factor * (
		    maxu * dwdxW + minu * dwdxE +
		    maxv * dwdyS + minv * dwdyN +
		    maxw * dwdzF + minw * dwdzB );

		uout[baseout + i] = uresult;
		vout[baseout + i] = vresult;
		wout[baseout + i] = wresult;
	    }
	}
    }
}
