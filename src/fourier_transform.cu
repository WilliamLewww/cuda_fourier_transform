#include <stdio.h>
#include <cmath>

__global__
void circularShift(float* dst, float* src, int width, int height, int shiftX, int shiftY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= width || idy >= height) { return; }

  dst[idy * width + idx] = src[(((idy - shiftY) % height + height) % height) * width + (((idx - shiftX) % width + width) % width)];
}

__global__
void fourierTransform(float* dst, float* src, int width, int height) {
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

extern "C" void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  float *h_image = (float*)malloc(width*height*sizeof(float));

  for (int x = 0; x < width * height * channels; x += channels) {
    h_image[x / channels] = ((src[x] * 0.30) + (src[x + 1] * 0.59) + (src[x + 2] * 0.11)) / 255.0;
  }

  float *d_fourierImage, *d_image;
  cudaMalloc(&d_fourierImage, width*height*sizeof(float));
  cudaMalloc(&d_image, width*height*sizeof(float));
  cudaMemcpy(d_image, h_image, width*height*sizeof(float), cudaMemcpyHostToDevice);

  fourierTransform<<<block, grid>>>(d_fourierImage, d_image, width, height);
  cudaDeviceSynchronize();

  float *d_circularFourierImage;
  cudaMalloc(&d_circularFourierImage, width*height*sizeof(float));
  circularShift<<<block, grid>>>(d_circularFourierImage, d_fourierImage, width, height, width / 2, height / 2);

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

    dst[x * channels] = h_circularFourierImage[x];
    dst[x * channels + 1] = h_circularFourierImage[x];
    dst[x * channels + 2] = h_circularFourierImage[x];
    dst[x * channels + 3] = 255;
  }

  cudaFree(d_circularFourierImage);
  cudaFree(d_fourierImage);
  cudaFree(d_image);
  free(h_circularFourierImage);
  free(h_image);
}