// exp_cordic.cpp
#include "exp_cordic.h"

// ---- LUT for atanh(2^-i), i=0..30 ----
static const double ATANH_2POW_NEG[] = {
    0.0,
    0.54930614433405484570,
    0.25541281188299534160,
    0.12565721414045303884,
    0.06258157147700301068,
    0.03126017849066699238,
    0.01562627175205221142,
    0.00781265895154042152,
    0.00390626986839682592,
    0.00195312748353254912,
    0.00097656281044103586,
    0.00048828128880511276,
    0.00024414062985063866,
    0.00012207031310632980,
    0.00006103515632579145,
    0.00003051757813447360,
    0.00001525878906368405,
    0.00000762939453139803,
    0.00000381469726564350,
    0.00000190734863281458,
    0.00000095367431640681,
    0.00000047683715820317,
    0.00000023841857910158,
    0.00000011920928955078,
    0.00000005960464477539,
    0.00000002980232238770,
    0.00000001490116119385,
    0.00000000745058059692,
    0.00000000372529029846,
    0.00000000186264514923,
    0.00000000093132257462
};

static inline cordic_z_t atanh_lut(int i) {
#pragma HLS INLINE
    return (cordic_z_t)ATANH_2POW_NEG[i];
}

// repeats within i<=30 are at i=4 and i=13
static inline bool is_repeat_index(int i) {
#pragma HLS INLINE
    return (i == 4) || (i == 13);
}

// Hyperbolic gain inverse (x0 = 1/K). Good approximation; keep for now.
static const double KINV_FOR_HYP_WITH_REPEATS = 1.207497067763072;


// Core hyperbolic CORDIC exp for inputs r in [0, ln2) (safe convergent region)
static exp_out_t exp_hcordic_core_small(cordic_z_t r) {
#pragma HLS INLINE

    cordic_z_t   z = r;  // residual
    cordic_vec_t x = (cordic_vec_t)KINV_FOR_HYP_WITH_REPEATS;
    cordic_vec_t y = (cordic_vec_t)0;

#if (CORDIC_STAGES > 30)
#error "Extend LUT/Kinv for CORDIC_STAGES > 30"
#endif

    for (int i = 1; i <= CORDIC_STAGES; ++i) {
#pragma HLS UNROLL
        {
            const bool d = (z >= 0);
            const cordic_vec_t x_sh = x >> i;
            const cordic_vec_t y_sh = y >> i;

            if (d) { x = x + y_sh; y = y + x_sh; z = z - atanh_lut(i); }
            else   { x = x - y_sh; y = y - x_sh; z = z + atanh_lut(i); }
        }

        if (is_repeat_index(i)) {
#pragma HLS UNROLL
            const bool d = (z >= 0);
            const cordic_vec_t x_sh = x >> i;
            const cordic_vec_t y_sh = y >> i;

            if (d) { x = x + y_sh; y = y + x_sh; z = z - atanh_lut(i); }
            else   { x = x - y_sh; y = y - x_sh; z = z + atanh_lut(i); }
        }
    }

    cordic_vec_t e = x + y;   // exp(r) = cosh(r)+sinh(r)
    if (e < 0) e = 0;
    return (exp_out_t)e;
}


// Top: exp(x) for x in [-8, 0]
exp_out_t exp_hcordic(exp_in_t xin) {
#pragma HLS INLINE off

    if (xin < (exp_in_t)-8.0) return (exp_out_t)0;
    if (xin > (exp_in_t) 0.0) xin = (exp_in_t)0.0;

    // Use wider type for range reduction.
    // Need to represent q = x/ln2 down to about -11.55 (for x=-8),
    // so give >=5 integer bits magnitude + sign -> choose 8 integer bits.
    typedef ap_fixed<24, 8> rr_t;

    const rr_t LN2     = (rr_t)0.6931471805599453094;
    const rr_t INV_LN2 = (rr_t)1.4426950408889634074; // 1/ln2

    rr_t xw = (rr_t)xin;

    // q = x/ln2 (avoid division; multiplication is safer in fixed-point)
    rr_t q = xw * INV_LN2;

    // k = floor(q) with correct behavior for negatives
    int k = (int)q;         // trunc toward 0
    if ((rr_t)k > q) k -= 1;

    // r = x - k*ln2  => should be in [0, ln2)
    rr_t r_w = xw - (rr_t)k * LN2;

    // Run core on r (keep as cordic_z_t to preserve precision)
    exp_out_t exp_r = exp_hcordic_core_small((cordic_z_t)r_w);

    // Scale by 2^k (k <= 0 here)
    if (k < 0) {
        exp_r = exp_r >> (-k);
    } else if (k > 0) {
        exp_r = exp_r << k;
    }

    return exp_r;
}
