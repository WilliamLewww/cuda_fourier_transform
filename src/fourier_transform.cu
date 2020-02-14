#include <stdio.h>
#include <cmath>

#define FOURIER_IMAGE_WIDTH 500
#define FOURIER_IMAGE_HEIGHT 500
#define FOURIER_IMAGE_CHANNELS 4

__global__
void luminosityGrayscale(float* dst, float* src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= FOURIER_IMAGE_WIDTH || idy >= FOURIER_IMAGE_HEIGHT) { return; }

  // 0.22 R + 0.72 G + 0.07 B
}

__global__
void circularShift(float* dst, float* src, int shiftX, int shiftY) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= FOURIER_IMAGE_WIDTH || idy >= FOURIER_IMAGE_HEIGHT) { return; }

  dst[idy * FOURIER_IMAGE_WIDTH + idx] = src[(((idy - shiftY) % FOURIER_IMAGE_HEIGHT + FOURIER_IMAGE_HEIGHT) % FOURIER_IMAGE_HEIGHT) * FOURIER_IMAGE_WIDTH + (((idx - shiftX) % FOURIER_IMAGE_WIDTH + FOURIER_IMAGE_WIDTH) % FOURIER_IMAGE_WIDTH)];
}

__global__
void fourierTransform(float* dst, float* src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= FOURIER_IMAGE_WIDTH || idy >= FOURIER_IMAGE_HEIGHT) { return; }

  float real = 0.0f;
  float imaginary = 0.0f;
  for (int y = 0; y < FOURIER_IMAGE_HEIGHT; y++) {
    for (int x = 0; x < FOURIER_IMAGE_WIDTH; x++) {
      real += src[y * FOURIER_IMAGE_WIDTH + x] * sinf((-2.0f * M_PI * idx * x / FOURIER_IMAGE_WIDTH) + (-2.0f * M_PI * idy * y / FOURIER_IMAGE_HEIGHT));
      imaginary += src[y * FOURIER_IMAGE_WIDTH + x] * cosf((-2.0f * M_PI * idx * x / FOURIER_IMAGE_WIDTH) + (-2.0f * M_PI * idy * y / FOURIER_IMAGE_HEIGHT));
    }
  }

  dst[idy * FOURIER_IMAGE_WIDTH + idx] = sqrtf((real * real) + (imaginary * imaginary));
}

extern "C" void fourierTransformWrapper(unsigned char* dst, unsigned char* src) {
  dim3 block(32, 32);
  dim3 grid((FOURIER_IMAGE_WIDTH + block.x - 1) / block.x, (FOURIER_IMAGE_HEIGHT + block.y - 1) / block.y);

  float *h_circularFourierImage = (float*)malloc(FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float));
  float *h_image = (float*)malloc(FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float));

  for (int x = 0; x < FOURIER_IMAGE_WIDTH * FOURIER_IMAGE_HEIGHT * FOURIER_IMAGE_CHANNELS; x += FOURIER_IMAGE_CHANNELS) {
    h_image[x / FOURIER_IMAGE_CHANNELS] = src[x] / 255.0;
  }

  float *d_fourierImage, *d_image;
  cudaMalloc(&d_fourierImage, FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float));
  cudaMalloc(&d_image, FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float));
  cudaMemcpy(d_image, h_image, FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float), cudaMemcpyHostToDevice);

  fourierTransform<<<block, grid>>>(d_fourierImage, d_image);
  cudaDeviceSynchronize();

  float *d_circularFourierImage;
  cudaMalloc(&d_circularFourierImage, FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float));
  circularShift<<<block, grid>>>(d_circularFourierImage, d_fourierImage, FOURIER_IMAGE_WIDTH / 2, FOURIER_IMAGE_HEIGHT / 2);

  cudaMemcpy(h_circularFourierImage, d_circularFourierImage, FOURIER_IMAGE_WIDTH*FOURIER_IMAGE_HEIGHT*sizeof(float), cudaMemcpyDeviceToHost);

  float max = 0.0;
  for (int x = 0; x < FOURIER_IMAGE_WIDTH * FOURIER_IMAGE_HEIGHT; x++) {
    if (h_circularFourierImage[x] > max) {
      max = h_circularFourierImage[x];
    }
  }

  float c = 255.0 / log(1 + fabs(max));
  for (int x = 0; x < FOURIER_IMAGE_WIDTH * FOURIER_IMAGE_HEIGHT; x++) {
    h_circularFourierImage[x] = c * log(1 + fabs(h_circularFourierImage[x]));

    dst[x * FOURIER_IMAGE_CHANNELS] = h_circularFourierImage[x];
    dst[x * FOURIER_IMAGE_CHANNELS + 1] = h_circularFourierImage[x];
    dst[x * FOURIER_IMAGE_CHANNELS + 2] = h_circularFourierImage[x];
    dst[x * FOURIER_IMAGE_CHANNELS + 3] = 255;
  }

  cudaFree(d_circularFourierImage);
  cudaFree(d_fourierImage);
  cudaFree(d_image);
  free(h_circularFourierImage);
  free(h_image);
}