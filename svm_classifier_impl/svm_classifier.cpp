#include "svm_classifier.h"
#include "exp_cordic.h"

extern double svs[NSV * DIM];     // [NSV*DIM]
extern double alphas[NSV];  // [NSV]
extern double bias[1];    // [1]

static const ap_fixed<18, 2> GAMMA = (ap_fixed<18, 2>)0.001;

// Clamp exp input to your exp_hcordic valid range [-8, 0]
static inline exp_in_t clamp_exp_in(exp_in_t v) {
#pragma HLS INLINE
    if (v < (exp_in_t)-8.0) return (exp_in_t)-8.0;
    if (v > (exp_in_t) 0.0) return (exp_in_t) 0.0;
    return v;
}

// Accessors (cast doubles to fixed-point)
static inline feat_t get_sv(int i, int j) {
#pragma HLS INLINE
    return (feat_t)svs[i*DIM + j];
}
static inline alpha_t get_alpha(int i) {
#pragma HLS INLINE
    return (alpha_t)alphas[i];
}
static inline bias_t get_bias() {
#pragma HLS INLINE
    return (bias_t)bias[0];
}

// L2 distance squared
static inline ap_fixed<32, 16> l2_dist2(const feat_t xbuf[DIM], int svi) {
#pragma HLS INLINE
    ap_fixed<32, 16> acc = 0;

    for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE II=1
        feat_t xi = get_sv(svi, j);
        ap_fixed<10, 8> d = (ap_fixed<10, 8>)(xi - xbuf[j]);
        ap_fixed<20, 16> d2 = (ap_fixed<20, 16>)(d * d);
        acc += (ap_fixed<32, 16>)d2;
    }
    return acc;
}

static inline ap_fixed<32, 12> svm_score_one(const feat_t xbuf[DIM]) {
#pragma HLS INLINE
    ap_fixed<32, 12> score = 0;

    for (int i = 0; i < NSV; ++i) {
#pragma HLS PIPELINE II=1
        ap_fixed<32, 16> dist2 = l2_dist2(xbuf, i);

        ap_fixed<32, 16> arg_fx = -(ap_fixed<32, 16>)(GAMMA * dist2);
        exp_in_t arg = clamp_exp_in((exp_in_t)arg_fx);

        exp_out_t k = exp_hcordic(arg);

        ap_fixed<32, 12> term = (ap_fixed<32, 12>)get_alpha(i) * (ap_fixed<32, 12>)k;
        score += term;
    }

    score += (ap_fixed<32, 12>)get_bias();
    return score;
}

// Top-level IP
void svm_classifier(const feat_t *x_images, pred_t *y_pred) {
#pragma HLS INTERFACE m_axi     port=x_images offset=slave bundle=gmem0 depth=2040576
#pragma HLS INTERFACE m_axi     port=y_pred   offset=slave bundle=gmem1 depth=2601
#pragma HLS INTERFACE s_axilite port=x_images bundle=control
#pragma HLS INTERFACE s_axilite port=y_pred   bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    feat_t xbuf[DIM];
#pragma HLS ARRAY_PARTITION variable=xbuf cyclic factor=8 dim=1

    for (int n = 0; n < NIMG; ++n) {

        for (int j = 0; j < DIM; ++j) {
#pragma HLS PIPELINE II=1
            xbuf[j] = x_images[n*DIM + j];
        }

        ap_fixed<32, 12> score = svm_score_one(xbuf);
        y_pred[n] = (score >= 0) ? (pred_t)1 : (pred_t)0;
    }
}
