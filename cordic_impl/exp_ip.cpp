/*#include "exp_ip.h"
#include <hls_math.h>

exp_out_t exp_ip(exp_in_t x) {
#pragma HLS INLINE off

    // Spec clamp: if x < -8, output 0
    if (x < (exp_in_t)-8.0) {
        return (exp_out_t)0;
    }

    // Temporary reference implementation (replace with your CORDIC later)
    // hls::exp works in C-sim/cosim; it’s fine as a placeholder to validate TB wiring.
    // Cast to float/double for the math, then back to fixed.
    float xf = (float)x;
    float yf = hls::expf(xf);

    return (exp_out_t)yf;
} */

#include "exp_ip.h"
#include "exp_cordic.h"

exp_out_t exp_ip(exp_in_t x) {
#pragma HLS INLINE off
    return exp_hcordic(x);
}
