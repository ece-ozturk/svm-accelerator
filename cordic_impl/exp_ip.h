// exp_ip.h
#pragma once
#include <ap_fixed.h>

// ---------------- Tuning knobs ----------------
// IMPORTANT:
// In this design, CORDIC_STAGES is the *maximum shift index i* (i = 1..CORDIC_STAGES).
// Hyperbolic CORDIC also repeats iterations at i = 4 and i = 13 (when those are within range).
// Therefore, the total number of pseudo-rotations is:
//   pseudo_rotations = CORDIC_STAGES + (CORDIC_STAGES>=4) + (CORDIC_STAGES>=13)
//
// To match the Python setting iters = 17 pseudo-rotations, use:
//   CORDIC_STAGES = 15  (15 + 1 + 1 = 17)
#ifndef CORDIC_STAGES
#define CORDIC_STAGES 15
#endif

// Input: x in [-8, 0]. Signed required.
// Q4.12: range [-8, +7.999], step 2^-12
typedef ap_fixed<16, 4> exp_in_t;

// Output: exp(x) in (0, 1]. Unsigned ok.
// Keep wide for accuracy first; shrink later if needed.
typedef ap_ufixed<24, 1> exp_out_t;

// Top function
exp_out_t exp_ip(exp_in_t x);
