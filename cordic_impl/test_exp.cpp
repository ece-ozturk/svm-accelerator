// test_exp.cpp
#include <cstdio>
#include <cstdint>
#include <cmath>

#include "exp_ip.h"

// ---------- Configuration ----------
static constexpr int    N_SAMPLES = 200000;
static constexpr double MSE_LIMIT = 2.4e-11;

// Quantize inputs on a Q4.12 grid (step = 2^-12)
static constexpr int    FRAC_BITS = 12;
static constexpr double STEP = 1.0 / (1u << FRAC_BITS);

// Uniform RNG (xorshift32) to avoid <random> variability across toolchains
static uint32_t rng_state = 0x12345678u;
static inline uint32_t xorshift32() {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

// Uniform in [0,1)
static inline double urand01() {
    // 24-bit fraction
    return (double)(xorshift32() & 0xFFFFFFu) / (double)(0x1000000u);
}

// Quantize to grid with saturation to [-8, 0]
static inline double quantize_to_step(double v) {
    if (v > 0.0) v = 0.0;
    if (v < -8.0) v = -8.0;

    double q = std::round(v / STEP) * STEP;

    if (q > 0.0) q = 0.0;
    if (q < -8.0) q = -8.0;
    return q;
}

int main() {
    // Print config (avoids stage/pseudo-rotation confusion)
    std::printf("CORDIC_STAGES (max shift i): %d\n", CORDIC_STAGES);
    std::printf("Pseudo-rotations (adds repeats at 4,13 if in range): %d\n\n",
                CORDIC_STAGES + (CORDIC_STAGES >= 4) + (CORDIC_STAGES >= 13));

    // ---- quick sanity checks ----
    {
        const double xs[] = { -8.0, -4.0, -1.0, -0.5, -0.04675, 0.0 };
        for (int i = 0; i < 6; ++i) {
            exp_in_t  xin = (exp_in_t)xs[i];
            exp_out_t y   = exp_ip(xin);
            double    ref = std::exp(xs[i]);
            std::printf("SANITY: x=% .6f  exp_ip=% .10f  ref=% .10f\n",
                        xs[i], (double)y, ref);
        }
        std::printf("\n");
    }

    // Randomized MSE test
    double sum_se  = 0.0;
    double sum_se2 = 0.0;

    for (int n = 0; n < N_SAMPLES; ++n) {
        double x;

        // Ensure endpoints appear in the sample set
        if ((n & 0x3FFF) == 0)      x = -8.0;
        else if ((n & 0x3FFF) == 1) x =  0.0;
        else                       x = -8.0 + 8.0 * urand01(); // uniform in [-8,0)

        x = quantize_to_step(x);

        exp_in_t  xin  = (exp_in_t)x;
        exp_out_t yfix = exp_ip(xin);

        double y_hat = (double)yfix;
        double y_ref = std::exp(x);

        double e  = (y_hat - y_ref);
        double se = e * e;

        sum_se  += se;
        sum_se2 += se * se;
    }

    const double mse = sum_se / (double)N_SAMPLES;

    // 95% CI for mean squared error (normal approx)
    const double mean_se  = mse;
    const double mean_se2 = sum_se2 / (double)N_SAMPLES;
    const double var_se   = (mean_se2 - mean_se * mean_se);
    const double se_mean  = std::sqrt(var_se / (double)N_SAMPLES);

    const double ci_half = 1.96 * se_mean;
    const double ci_lo   = mean_se - ci_half;
    const double ci_hi   = mean_se + ci_half;

    std::printf("=== exp_ip testbench ===\n");
    std::printf("Samples         : %d\n", N_SAMPLES);
    std::printf("Input quant step: %.12g (2^-%d)\n", STEP, FRAC_BITS);
    std::printf("MSE             : %.16e\n", mse);
    std::printf("95%% CI (approx) : [%.16e, %.16e]\n", ci_lo, ci_hi);
    std::printf("Spec MSE limit  : %.16e\n", MSE_LIMIT);

    if (ci_hi < MSE_LIMIT) {
        std::printf("PASS: CI upper < limit\n");
        return 0;
    } else {
        std::printf("FAIL: CI upper >= limit\n");
        return 1;
    }
}
