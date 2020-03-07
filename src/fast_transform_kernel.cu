#include <cuComplex.h>
#include <stdio.h>
#include <complex.h>

cuFloatComplex cuexpf(cuFloatComplex value) {
  float exponent = expf(value.x);
  float real, imaginary;
  sincosf(value.y, &imaginary, &real);

  return make_cuFloatComplex(real * exponent, imaginary * exponent);
}

__global__
void recursiveFastFourierTransform(cuFloatComplex* dst, cuFloatComplex* src, int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
}

extern "C" void fastFourierTransformWrapper2D(float* dst, float* src, int width, int height) {
  dim3 block(32, 32);
  dim3 grid((block.x + width - 1) / block.x, (block.y + height - 1) / block.y);

  cuFloatComplex* h_buffer = (cuFloatComplex*)malloc(width*height*sizeof(cuFloatComplex));
  for (int x = 0; x < width * height; x++) { h_buffer[x] = make_cuFloatComplex(src[x], 0.0); }

  cuFloatComplex* d_buffer;
  cudaMalloc(&d_buffer, width*height*sizeof(cuFloatComplex));
  cudaMemcpy(d_buffer, h_buffer, width*height*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

  cuFloatComplex* d_bufferResult;
  cudaMalloc(&d_bufferResult, width*height*sizeof(cuFloatComplex));

  recursiveFastFourierTransform<<<grid, block>>>(d_bufferResult, d_buffer, width, height);

  cudaDeviceReset();
}