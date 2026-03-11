#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "c_headers/alphas.h"
#include "c_headers/bias.h"
#include "c_headers/ground_truth.h"
#include "c_headers/svs.h"
#include "c_headers/test_data.h"
#include "svm_classifier.h"

#define IMG_SIZE 784 // Each image is 28x28
#define NUM_IMGS 2601

#define NSV 165

double classify(double x[IMG_SIZE]);

int main() {
  // for debugging
  std::ofstream scoresF;
  scoresF.open("scores.txt");

  double x[IMG_SIZE];
  double scores[NUM_IMGS];
  for (int i = 0; i < NUM_IMGS; i++) {
    // form input vector x
    for (int j = 0; j < IMG_SIZE; j++)
      x[j] = test_data[i * IMG_SIZE + j];

    // call the function
    scores[i] = classify(x);

    // store scores to file for debugging
    scoresF << scores[i] << std::endl;
  }
  scoresF.close();

  // ---- Run the HLS IP ----
  packed_feat_t *hw_x_images =
      (packed_feat_t *)malloc(NUM_IMGS * (DIM / 8) * sizeof(packed_feat_t));
  for (int i = 0; i < NUM_IMGS; i++) {
    for (int j = 0; j < DIM; j += 8) {
      packed_feat_t pack = 0;
      for (int k = 0; k < 8; ++k) {
        feat_t f = (feat_t)test_data[i * IMG_SIZE + j + k];
        pack.range(k * 8 + 7, k * 8) = f.range();
      }
      hw_x_images[i * (DIM / 8) + (j / 8)] = pack;
    }
  }

  pred_t *hw_predictions = (pred_t *)malloc(NUM_IMGS * sizeof(pred_t));
  svm_classifier(hw_x_images, hw_predictions);

  // get predictions --> this takes the sign() of the output
  int predictions[NUM_IMGS];
  for (int i = 0; i < NUM_IMGS; i++) {
    // classifying between 0 and 1
    if (scores[i] < 0)
      predictions[i] = 0;
    else
      predictions[i] = 1;
  }

  // summary statistics
  double accuracy = 0.0;
  int correct = 0;
  for (int i = 0; i < NUM_IMGS; i++) {
    if (predictions[i] == ground_truth[i])
      correct++;
  }
  accuracy = correct / double(NUM_IMGS);
  printf("Reference Float Classification Accuracy: %f\n", accuracy);

  // hw summary statistics
  double hw_accuracy = 0.0;
  int hw_correct = 0;
  for (int i = 0; i < NUM_IMGS; i++) {
    if ((int)hw_predictions[i] == ground_truth[i])
      hw_correct++;
  }
  hw_accuracy = hw_correct / double(NUM_IMGS);
  printf("Hardware Fixed-Point Classification Accuracy: %f\n", hw_accuracy);

  free(hw_x_images);
  free(hw_predictions);

  // summary statistics - confusion matrix
  double CM[2][2];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      CM[i][j] = 0.0;

  for (int i = 0; i < NUM_IMGS; i++) {
    CM[ground_truth[i]][predictions[i]]++;
  }
  printf("Confusion Matrix (%d test points):\n", NUM_IMGS);
  printf("%f, %f\n", CM[0][0] / NUM_IMGS, CM[0][1] / NUM_IMGS);
  printf("%f, %f\n", CM[1][0] / NUM_IMGS, CM[1][1] / NUM_IMGS);
}

double classify(double x[IMG_SIZE]) {
  double sum = 0.0;
  for (int i = 0; i < NSV; i++) {
    double alpha = alphas[i];

    // rbf kernel
    double l2Squared = 0.0;
    for (int j = 0; j < IMG_SIZE; j++) {
      double _sv = svs[i * IMG_SIZE + j];
      double _x = x[j];
      l2Squared += (_sv - _x) * (_sv - _x);
    }
    double K = exp(-0.001 * l2Squared);

    sum += alpha * K;
  }
  sum = sum + bias[0];
  return sum;
}
