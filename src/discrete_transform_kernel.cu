#include <stdio.h>
#include <cmath>

__global__
void circularShift2D(float* dst, float* src, int width, int height, int shiftX, int shiftY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= width || idy >= height) { return; }

  dst[idy * width + idx] = src[(((idy - shiftY) % height + height) % height) * width + (((idx - shiftX) % width + width) % width)];
}

__global__
void discreteFourierTransform2D(float* dst, float* src, int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= width || idy >= height) { return; }

  float real = 0.0f;
  float imaginary = 0.0f;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      real += src[y * width + x] * sinf((-2.0f * M_PI * idx * x / width) + (-2.0f * M_PI * idy * y / height));
      imaginary += src[y * width + x] * cosf((-2.0f * M_PI * idx * x / width) + (-2.0f * M_PI * idy * y / height));
    }
  }

  dst[idy * width + idx] = sqrtf((real * real) + (imaginary * imaginary));
}

extern "C" void discreteFourierTransformWrapper2D(float* dst, float* src, int width, int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  float *d_fourierImage, *d_image;
  cudaMalloc(&d_fourierImage, width*height*sizeof(float));
  cudaMalloc(&d_image, width*height*sizeof(float));
  cudaMemcpy(d_image, src, width*height*sizeof(float), cudaMemcpyHostToDevice);

  discreteFourierTransform2D<<<block, grid>>>(d_fourierImage, d_image, width, height);
  cudaDeviceSynchronize();

  float *d_circularFourierImage;
  cudaMalloc(&d_circularFourierImage, width*height*sizeof(float));
  circularShift2D<<<block, grid>>>(d_circularFourierImage, d_fourierImage, width, height, width / 2, height / 2);

  float *h_circularFourierImage = (float*)malloc(width*height*sizeof(float));
  cudaMemcpy(h_circularFourierImage, d_circularFourierImage, width*height*sizeof(float), cudaMemcpyDeviceToHost);

  float max = 0.0;
  for (int x = 0; x < width * height; x++) {
    if (h_circularFourierImage[x] > max) {
      max = h_circularFourierImage[x];
    }
  }

  float c = 255.0 / log(1 + fabs(max));
  for (int x = 0; x < width * height; x++) {
    h_circularFourierImage[x] = c * log(1 + fabs(h_circularFourierImage[x]));
    dst[x] = h_circularFourierImage[x];
  }

  cudaFree(d_circularFourierImage);
  cudaFree(d_fourierImage);
  cudaFree(d_image);
  free(h_circularFourierImage);
}

__global__
void circularShiftBatch2D(float* dst, float* src, int width, int height, int depth, int shiftX, int shiftY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= width || idy >= height || idz >= depth) { return; }

  dst[idz * width * height + idy * width + idx] = src[idz * width * height + (((idy - shiftY) % height + height) % height) * width + (((idx - shiftX) % width + width) % width)];
}

__global__
void discreteFourierTransformBatch2D(float* dst, float* src, int width, int height, int depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx >= width || idy >= height || idz >= depth) { return; }

  float real = 0.0f;
  float imaginary = 0.0f;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      real += src[idz * width * height + y * width + x] * sinf((-2.0f * M_PI * idx * x / width) + (-2.0f * M_PI * idy * y / height));
      imaginary += src[idz * width * height + y * width + x] * cosf((-2.0f * M_PI * idx * x / width) + (-2.0f * M_PI * idy * y / height));
    }
  }

  dst[idz * width * height + idy * width + idx] = sqrtf((real * real) + (imaginary * imaginary));
}

extern "C" void discreteFourierTransformBatchWrapper2D(float* dst, float* src, int width, int height, int depth) {
  dim3 block(32, 32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, (depth + block.z - 1) / block.z);

  float *d_fourierImage, *d_image;
  cudaMalloc(&d_fourierImage, width*height*depth*sizeof(float));
  cudaMalloc(&d_image, width*height*depth*sizeof(float));
  cudaMemcpy(d_image, src, width*height*depth*sizeof(float), cudaMemcpyHostToDevice);

  discreteFourierTransformBatch2D<<<block, grid>>>(d_fourierImage, d_image, width, height, depth);
  cudaDeviceSynchronize();

  float *d_circularFourierImage;
  cudaMalloc(&d_circularFourierImage, width*height*depth*sizeof(float));
  circularShiftBatch2D<<<block, grid>>>(d_circularFourierImage, d_fourierImage, width, height, depth, width / 2, height / 2);

  float *h_circularFourierImage = (float*)malloc(width*height*depth*sizeof(float));
  cudaMemcpy(h_circularFourierImage, d_circularFourierImage, width*height*depth*sizeof(float), cudaMemcpyDeviceToHost);

  float max = 0.0;
  for (int x = 0; x < width * height * depth; x++) {
    if (h_circularFourierImage[x] > max) {
      max = h_circularFourierImage[x];
    }
  }

  float c = 255.0 / log(1 + fabs(max));
  for (int x = 0; x < width * height * depth; x++) {
    h_circularFourierImage[x] = c * log(1 + fabs(h_circularFourierImage[x]));
    dst[x] = h_circularFourierImage[x];
  }

  cudaFree(d_circularFourierImage);
  cudaFree(d_fourierImage);
  cudaFree(d_image);
  free(h_circularFourierImage);
}