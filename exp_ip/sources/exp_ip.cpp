/* exp_ip.cpp*/
#include "exp_ip.h"
#include "cordic.h"
#include <cmath>

data_t quantise(input_t z)
{
    if (z < SAT_THRESHOLD) { return 0; }
    return (data_t)z;
}

void range_reduce(data_t z, int &k, data_t &r)
{
    k = (int)floor((double)z / (double)LN2);
    r = z - k * LN2;
}

output_t exp_ip(input_t z)
{
    int k;
    data_t r;
    data_t z_q = quantise(z);

    if (z_q == 0) { return 0; }

    range_reduce(z_q, k, r);
    output_t result = cordic(r);
    return result >> -k;
}

