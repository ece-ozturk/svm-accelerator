// =========================
// exp_cordic.cpp
// =========================
#include "exp_cordic.h"

static const cordic_z_t ATANH_2POW_NEG[] = {
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
    0.00000095367431640681
};

static inline cordic_z_t atanh_lut(int i) {
#pragma HLS INLINE
    return ATANH_2POW_NEG[i];
}

// repeats at i=4 and i=13
static inline bool is_repeat_index(int i) {
#pragma HLS INLINE
    return (i == 4) || (i == 13);
}

// IMPORTANT FIX:
// Keep Kinv as a fixed-point constant (not double -> cast),
// to avoid systematic quantization bias.
// (This value corresponds to hyperbolic CORDIC with repeats at 4 and 13.)
static const cordic_vec_t KINV_FOR_HYP_WITH_REPEATS = cordic_vec_t(1.207497067763072);

// Core CORDIC for r in [0, ln2)
static exp_out_t exp_hcordic_core_small(cordic_z_t r) {
#pragma HLS INLINE

    cordic_z_t   z = r;
    cordic_vec_t x = KINV_FOR_HYP_WITH_REPEATS;
    cordic_vec_t y = 0;

    for (int i = 1; i <= CORDIC_STAGES; ++i) {
#pragma HLS UNROLL

        const bool d = (z >= 0);
        const cordic_vec_t x_sh = x >> i;
        const cordic_vec_t y_sh = y >> i;

        if (d) { x = x + y_sh; y = y + x_sh; z = z - atanh_lut(i); }
        else   { x = x - y_sh; y = y - x_sh; z = z + atanh_lut(i); }

        // repeat stage explicitly
        if (i == 4 || i == 13) {

            const bool d2 = (z >= 0);
            const cordic_vec_t x_sh2 = x >> i;
            const cordic_vec_t y_sh2 = y >> i;

            if (d2) { x = x + y_sh2; y = y + x_sh2; z = z - atanh_lut(i); }
            else    { x = x - y_sh2; y = y - x_sh2; z = z + atanh_lut(i); }
        }
    }

    cordic_vec_t e = x + y;
    if (e < 0) e = 0;

    // rounding for Q4.17
    const cordic_vec_t rounding = cordic_vec_t(1.0 / (1 << 18));
    e += rounding;

    return (exp_out_t)e;
}

// Top: exp(x) for x in [-8,0]
exp_out_t exp_hcordic(exp_in_t xin) {
#pragma HLS INLINE off

    if (xin < (exp_in_t)-8.0) return (exp_out_t)0;
    if (xin > (exp_in_t) 0.0) xin = (exp_in_t)0.0;

    // IMPORTANT FIX:
    // Use higher precision for range reduction so its error doesn't dominate Q4.16 output.
    typedef ap_fixed<32, 8> rr_t;

    const rr_t LN2     = (rr_t)0.6931471805599453094;
    const rr_t INV_LN2 = (rr_t)1.4426950408889634074;

    rr_t xw = (rr_t)xin;

    rr_t q = xw * INV_LN2;

    int k = (int)q;         // trunc toward 0
    if ((rr_t)k > q) k -= 1;

    rr_t r_w = xw - (rr_t)k * LN2;

    exp_out_t exp_r = exp_hcordic_core_small((cordic_z_t)r_w);

    if (k < 0)      exp_r = exp_r >> (-k);
    else if (k > 0) exp_r = exp_r << k;

    return exp_r;
}
