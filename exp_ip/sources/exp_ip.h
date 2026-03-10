/* exp_ip.h - Header file for exponential calculating ip*/
#ifndef EXP_IP_H
#define EXP_IP_H
#include "ap_fixed.h"

// Assign fixed-point data types for values within CORDIC algorithm:
typedef ap_fixed<20, 4> data_t; 		// Input data type with precision calculated by Monte-Carlo Simulation
typedef ap_fixed<21, 5> output_t; 		// Output data type, slightly wider to account for overflow
typedef ap_fixed<36, 4> internal_t; 	// Internal CORDIC precision, currently accounting for maximum value due to bit shifts
typedef float input_t; 					// exp_ip expects single-precision as provided by specifications

const input_t SAT_THRESHOLD = -8;
const data_t LN2_1 = 0.69314718055994530942;
const data_t LN2_2  = 1.38629436;
const data_t LN2_3  = 2.07944154;
const data_t LN2_4  = 2.77258872;
const data_t LN2_5  = 3.46573590;
const data_t LN2_6  = 4.15888308;
const data_t LN2_7  = 4.85203026;
const data_t LN2_8  = 5.54517744;
const data_t LN2_9  = 6.23832462;
const data_t LN2_10 = 6.93147180;
const data_t LN2_11 = 7.62461899;
const data_t LN2_12 = 8.31776617;

data_t quantise(input_t z);
void range_reduce(data_t z, int &k, data_t &r);
output_t exp_ip(input_t z);

#endif
