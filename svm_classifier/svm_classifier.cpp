#include "svm_classifier.h"
#include "exp_piecewise.h"

// ------------------------------------------------------------
// Model source selection
// ------------------------------------------------------------
#ifdef __SYNTHESIS__
#include "svm_model_fixed.h" // provides: svs_q[NSV][DIM], alphas_q[NSV], bias_q[1]
#else
extern double svs[NSV * DIM];
extern double alphas[NSV];
extern double bias[1];
#endif

// ------------------------------------------------------------
// Tunables
// ------------------------------------------------------------
// Feature unroll factor — matched to memory banking below.
static const int U = 16;

// Support vectors processed in parallel by the distance engine
static const int SV_PAR = 8;

static const int SV_GROUPS = (NSV + SV_PAR - 1) / SV_PAR;

static const ap_fixed<18, 2> GAMMA = (ap_fixed<18, 2>)0.001;

// Internal accumulator types — narrowed for better resource usage
typedef ap_fixed<24, 14> dist_acc_t;
typedef ap_fixed<24, 10> score_t;

// ------------------------------------------------------------
// Clamp exp input to exp_hcordic valid range [-8, 0]
// ------------------------------------------------------------
static inline exp_in_t clamp_exp_in(exp_in_t v) {
#pragma HLS INLINE
  if (v < (exp_in_t)-8.0)
    return (exp_in_t)-8.0;
  if (v > (exp_in_t)0.0)
    return (exp_in_t)0.0;
  return v;
}

// ------------------------------------------------------------
// Accessors for CSIM / SYNTHESIS
// ------------------------------------------------------------
static inline feat_t get_sv(int i, int j) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  return svs_q[i][j];
#else
  return (feat_t)svs[i * DIM + j];
#endif
}

static inline alpha_t get_alpha(int i) {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  return alphas_q[i];
#else
  return (alpha_t)alphas[i];
#endif
}

static inline bias_t get_bias() {
#pragma HLS INLINE
#ifdef __SYNTHESIS__
  return bias_q[0];
#else
  return (bias_t)bias[0];
#endif
}

// ------------------------------------------------------------
// Distance engine
// Computes SV_PAR squared L2 distances in parallel with U lanes
// ------------------------------------------------------------
static void l2_dist2_u8_sv8(const feat_t xbuf[DIM], int base_svi,
                            dist_acc_t dist2[SV_PAR]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = dist2 complete dim = 1

  // Initialize accumulators
  for (int p = 0; p < SV_PAR; ++p) {
#pragma HLS UNROLL
    dist2[p] = 0;
  }

  // DIM = 784, U = 16 -> 49 iterations
  for (int j = 0; j < DIM; j += U) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 49 max = 49

    for (int p = 0; p < SV_PAR; ++p) {
#pragma HLS UNROLL
      const int svi = base_svi + p;

      if (svi < NSV) {
        for (int u = 0; u < U; ++u) {
#pragma HLS UNROLL
          feat_t xi = get_sv(svi, j + u);
          feat_t xx = xbuf[j + u];

          ap_fixed<9, 8> d = (ap_fixed<9, 8>)(xi - xx);
          ap_fixed<18, 16> d2 = (ap_fixed<18, 16>)(d * d);

          dist2[p] += (dist_acc_t)d2;
        }
      }
    }
  }
}

// ------------------------------------------------------------
// Score one image
// ------------------------------------------------------------
static score_t svm_score_one(const feat_t xbuf[DIM]) {
#pragma HLS INLINE off

  score_t score = 0;

  // exp_piecewise is heavily inlined and parallelized
  for (int g = 0; g < SV_GROUPS; ++g) {
#pragma HLS LOOP_TRIPCOUNT min = 21 max = 21

    const int base_svi = g * SV_PAR;

    dist_acc_t dist2[SV_PAR];
#pragma HLS ARRAY_PARTITION variable = dist2 complete dim = 1

    l2_dist2_u8_sv8(xbuf, base_svi, dist2);

    // Kernel + weighted accumulation
    for (int p = 0; p < SV_PAR; ++p) {
#pragma HLS UNROLL
      const int svi = base_svi + p;

      if (svi < NSV) {
        ap_fixed<24, 14> arg_fx = -(ap_fixed<24, 14>)(GAMMA * dist2[p]);
        exp_in_t arg = clamp_exp_in((exp_in_t)arg_fx);

        exp_out_t k = exp_piecewise(arg);

        score_t term = (score_t)get_alpha(svi) * (score_t)k;
        score += term;
      }
    }
  }

  score += (score_t)get_bias();
  return score;
}

// ------------------------------------------------------------
// Top-level IP
// ------------------------------------------------------------
void svm_classifier(const packed_feat_t *x_images, pred_t *y_pred) {
#pragma HLS INTERFACE m_axi port = x_images offset = slave bundle =            \
    gmem0 depth = 254898
#pragma HLS INTERFACE m_axi port = y_pred offset = slave bundle =              \
    gmem1 depth = 2601

#pragma HLS INTERFACE s_axilite port = x_images bundle = control
#pragma HLS INTERFACE s_axilite port = y_pred bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#ifdef __SYNTHESIS__
  // Coursework recommendation: keep model parameters in PL BRAM/ROM
#pragma HLS RESOURCE variable = svs_q core = ROM_nP_BRAM
#pragma HLS RESOURCE variable = alphas_q core = ROM_nP_BRAM

  // Bank SV feature dimension for U=16 access pattern
#pragma HLS ARRAY_PARTITION variable = svs_q cyclic factor = 16 dim = 2
#endif

  // Local buffer for one image
  feat_t xbuf[DIM];
#pragma HLS ARRAY_PARTITION variable = xbuf cyclic factor = 16 dim = 1

  for (int n = 0; n < NIMG; ++n) {
#pragma HLS LOOP_TRIPCOUNT min = 2601 max = 2601
#pragma HLS DATAFLOW

    // Load one image from DDR (8 features per 64-bit beat)
    for (int j = 0; j < DIM / 8; ++j) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 98 max = 98
      packed_feat_t pack = x_images[n * (DIM / 8) + j];
      for (int k = 0; k < 8; ++k) {
#pragma HLS UNROLL
        feat_t f;
        f.range() = pack.range(k * 8 + 7, k * 8);
        xbuf[j * 8 + k] = f;
      }
    }

    score_t score = svm_score_one(xbuf);
    y_pred[n] = (score >= 0) ? (pred_t)1 : (pred_t)0;
  }
}
