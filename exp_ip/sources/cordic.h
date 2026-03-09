/* cordic.h - Header file for CORDIC algorithm*/
#ifndef CORDIC_H
#define CORDIC_H

#include "ap_fixed.h"
#include "exp_ip.h"

// Initialise value for 1/K
const internal_t K_n = 1.20749706771621512225;

// Function declaration
output_t cordic(data_t z);

#endif
