__global__
void fourierTransform(unsigned char* dst, unsigned char* src, int width, int height) {
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (idx >= width || idy >= height) { return; }
}

extern "C" void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height) {
  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  unsigned char *d_dst, *d_src;
  cudaMalloc(&d_dst, width*height*sizeof(unsigned char));
  cudaMalloc(&d_src, width*height*sizeof(unsigned char));
  cudaMemcpy(d_src, src, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

  fourierTransform<<<block, grid>>>(d_dst, d_src, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(dst, d_dst, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(d_dst);
  cudaFree(d_src);
}