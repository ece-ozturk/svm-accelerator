// =========================
// exp_ip.cpp
// =========================
#include "exp_ip.h"
#include "exp_piecewise.h"

exp_out_t exp_ip(exp_in_t x) {
#pragma HLS INLINE off
  return exp_piecewise(x);
}
