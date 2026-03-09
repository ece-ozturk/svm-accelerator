/* exp_ip.h - Header file for exponential calculating ip*/
#ifndef EXP_IP_H
#define EXP_IP_H
#include "ap_fixed.h"

// Assign fixed-point data types for values within CORDIC algorithm:
typedef ap_fixed<20, 4> data_t; 		// Input data type with precision calculated by Monte-Carlo Simulation
typedef ap_fixed<21, 5> output_t; 		// Output data type, slightly wider to account for overflow
typedef ap_fixed<36, 4> internal_t; 	// Internal CORDIC precision, currently accounting for maximum value due to bit shifts
typedef ap_fixed<28, 15> input_t; 		// Input to exp_ip

const data_t SAT_THRESHOLD = -8;
const data_t LN2 = 0.69314718055994530942;

data_t quantise(input_t z);
void range_reduce(data_t z, int &k, data_t &r);
output_t exp_ip(input_t z);

#endif
