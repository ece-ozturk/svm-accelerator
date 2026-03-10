/* exp_ip.cpp
 * Top-level file for IP that evaluates exp(x)
 * quantise(): 		Quantises input to fixed point with precision evaluated by Monte-Carlo simulation
 * range_reduce():  Transforms input into convergent range for CORDIC algorithm
 * exp_ip():		Main function, evaluating exp(x) using CORDIC */
/* -------------------------------------------------------------------------------------- */
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
    if      (z >= 0)        { k = 0;   r = z; }
    else if (z >= -LN2_1)   { k = -1;  r = z + LN2_1; }
    else if (z >= -LN2_2)   { k = -2;  r = z + LN2_2; }
    else if (z >= -LN2_3)   { k = -3;  r = z + LN2_3; }
    else if (z >= -LN2_4)   { k = -4;  r = z + LN2_4; }
    else if (z >= -LN2_5)   { k = -5;  r = z + LN2_5; }
    else if (z >= -LN2_6)   { k = -6;  r = z + LN2_6; }
    else if (z >= -LN2_7)   { k = -7;  r = z + LN2_7; }
    else if (z >= -LN2_8)   { k = -8;  r = z + LN2_8; }
    else if (z >= -LN2_9)   { k = -9;  r = z + LN2_9; }
    else if (z >= -LN2_10)  { k = -10; r = z + LN2_10; }
    else if (z >= -LN2_11)  { k = -11; r = z + LN2_11; }
    else                    { k = -12; r = z + LN2_12; }
}

/* void range_reduce(data_t z, int &k, data_t &r)
{
    k = (int)floor((double)z / (double)LN2);
    r = z - k * LN2;
} */

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

