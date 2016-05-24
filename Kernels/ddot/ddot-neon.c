#include <arm_neon.h>

#define vtype float64x2_t
#define vtype_2 float64x2x2_t
#define PREFETCH __builtin_prefetch
#define LOAD  vld1q_f64
#define STORE vst1q_f64

vtype set_vector(double val)
{
        vtype ret;
        ret = vsetq_lane_f64(val, ret, 0);
        ret = vsetq_lane_f64(val, ret, 1);
        return ret;
}

double ddot(const int N, const double *a, const int incx, const double *b, const int incy)
{
	int i;
	
	vtype q00 = set_vector(0.); 
	vtype q01 = set_vector(0.); 
	vtype q0a, q1a;
	vtype q0b, q1b;
	//
	double c;
	//
	for (i = 0; i < N - N%4; i = i + 4)
	{
		q0a = LOAD(a + i);
		q0b = LOAD(b + i);
		q00 = vfmaq_f64(q00, q0a, q0b);	
		//q0a = vmulq_f64(q0a, q0b);
		//q00 = vaddq_f64(q0a, q00);
		//
		q0a = LOAD(a + i + 2);
                q1b = LOAD(b + i + 2);
                q01 = vfmaq_f64(q01, q0a, q0b);
                //q1a = vmulq_f64(q1a, q1b);
                //q01 = vaddq_f64(q1a, q01);
		//c += a [i]*b [i];
	}
	c = q00[0] + q00[1] + q01[0] + q01[1];
	return c;
}
