#pragma once

#include <ap_fixed.h>
#include <ap_int.h>

// Dataset sizes
static const int DIM  = 784;   // 28*28
static const int NSV  = 165;
static const int NIMG = 2601;

// Quantized types per coursework guide
typedef ap_fixed<8, 7>  feat_t;   // xi, x : 7 int, 1 frac
typedef ap_fixed<8, 5>  alpha_t;  // yi*ai : 5 int, 3 frac
typedef ap_fixed<8, 1>  bias_t;   // b : 1 int, 7 frac

// Prediction output
typedef ap_uint<1> pred_t;

// Top-level IP
void svm_classifier(
    const feat_t *x_images,  // [NIMG*DIM]
    pred_t       *y_pred     // [NIMG]
);
