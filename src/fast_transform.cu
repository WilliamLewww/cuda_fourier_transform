#include <stdio.h>
#include <cmath>

float* recursiveFourierTransformCPU(float* samples, int size) {
  if (size == 1) {
    return samples;
  }

  float* even = (float*)malloc(size/2*sizeof(float));
  float* odd = (float*)malloc(size/2*sizeof(float));

  for (int x = 0; x < size / 2; x++) {
    even[x] = samples[2 * x];
    odd[x] = samples[2  * x + 1];
  }

  float* fEven = recursiveFourierTransformCPU(even, size / 2);
  float* fOdd = recursiveFourierTransformCPU(odd, size / 2);

  float* bins = (float*)malloc(size*sizeof(float));
  for (int x = 0; x < size / 2; x++) {
    float real = sinf(-2.0 * M_PI * x / size) * fOdd[x];
    float imaginary = cosf(-2.0 * M_PI * x / size) * fOdd[x];

    bins[x] = fEven[x] + sqrtf((real * real) + (imaginary * imaginary));
    bins[x + (size / 2)] = fEven[x] - sqrtf((real * real) + (imaginary * imaginary));
  }

  return bins;
}

extern "C" void fastFourierTransformCPU(float* dst, float* src, int width, int height) {
  float* samples = (float*)malloc(width*height*sizeof(float));
  memcpy(samples, src, width*height*sizeof(float));

  float* fourierSamples;
  for (int row = 0; row < height; row++) {
    fourierSamples = recursiveFourierTransformCPU(&samples[row * width], width);
    memcpy(&dst[row * width], fourierSamples, width*sizeof(float));
  }
}