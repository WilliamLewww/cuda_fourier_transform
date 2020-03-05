#include <cuComplex.h>
#include <stdio.h>
#include <complex.h>

cuFloatComplex cuexpf(cuFloatComplex value) {
  float exponent = expf(value.x);
  float real, imaginary;
  sincosf(value.y, &imaginary, &real);

  return make_cuFloatComplex(real * exponent, imaginary * exponent);
}

extern "C" void fastFourierTransformWrapper2D(float* dst, float* src, int width, int height) {
  
}