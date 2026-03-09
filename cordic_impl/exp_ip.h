// =========================
// exp_ip.h
// =========================
#pragma once
#include <ap_fixed.h>

// CORDIC shift-index upper bound (i = 1..CORDIC_STAGES).
// Hyperbolic CORDIC repeats at i=4 and i=13 if within range.
#define CORDIC_STAGES 16

// Input: represent [-8,0] on a Q4.12-ish grid (12 frac bits)
typedef ap_fixed<16, 4> exp_in_t;

// Output: Q4.17 (21 bits total)
typedef ap_fixed<21, 4> exp_out_t;

// Top-level exp IP function
exp_out_t exp_ip(exp_in_t x);
