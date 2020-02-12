#include <stdio.h>

__global__
void fourierTransform(float* dst, float* src, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= width || idy >= height) { return; }

  float sum = 0.0;
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      sum += src[y * width + x] * expf(-idy * 2.0 * M_PI * (((idx * x) / width) + ((idy * y) / height)));
    }
  }

  dst[idy * width + idx] = sum;
}

extern "C" void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  float *h_dst = (float*)malloc(width*height*sizeof(float));
  float *h_src = (float*)malloc(width*height*sizeof(float));

  for (int x = 0; x < width * height * channels; x += channels) {
    h_src[x / channels] = src[x];
  }

  float *d_dst, *d_src;
  cudaMalloc(&d_dst, width*height*sizeof(float));
  cudaMalloc(&d_src, width*height*sizeof(float));
  cudaMemcpy(d_src, h_src, width*height*sizeof(float), cudaMemcpyHostToDevice);

  fourierTransform<<<block, grid>>>(d_dst, d_src, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(h_dst, d_dst, width*height*sizeof(float), cudaMemcpyDeviceToHost);

  for (int x = 0; x < width * height; x++) {
    dst[x * channels] = h_dst[x] * 255;
    dst[x * channels + 1] = h_dst[x] * 255;
    dst[x * channels + 2] = h_dst[x] * 255;
    dst[x * channels + 3] = h_dst[x] * 255;
  }

  cudaFree(d_dst);
  cudaFree(d_src);
  free(h_dst);
  free(h_src);
}