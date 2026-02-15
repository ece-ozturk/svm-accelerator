#pragma once
#include "exp_ip.h"
#include <ap_fixed.h>

// Vector path (x,y): needs enough integer bits to hold ~1.2 comfortably
typedef ap_fixed<32, 3> cordic_vec_t;

// Residual z path: give extra integer headroom for accumulation of atanh terms
typedef ap_fixed<32, 3> cordic_z_t;

exp_out_t exp_hcordic(exp_in_t x);
