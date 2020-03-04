#include <stdio.h>
#include <cmath>
#include <complex>

void recursiveFastFourierTransformCPU(std::complex<float>* bufferCombine, std::complex<float>* bufferSplit, int size, int stride) {
  if (stride < size) {
    recursiveFastFourierTransformCPU(bufferSplit, bufferCombine, size, stride * 2);
    recursiveFastFourierTransformCPU(bufferSplit + stride, bufferCombine + stride, size, stride * 2);
 
    for (int i = 0; i < size; i += 2 * stride) {
      std::complex<float> t = std::exp(-std::complex<float>(0, 1) * float(M_PI) * float(i) / float(size)) * bufferSplit[i + stride];
      bufferCombine[i / 2] = bufferSplit[i] + t;
      bufferCombine[(i + size)/2] = bufferSplit[i] - t;
    }
  }
}

void fastFourierTransformCPU(float* dst, float* src, int width, int height) {
  std::complex<float>* imageBuffer = (std::complex<float>*)malloc(width*height*sizeof(std::complex<float>));

  std::complex<float>* buffer = (std::complex<float>*)malloc(width*sizeof(std::complex<float>));
  std::complex<float>* bufferClone = (std::complex<float>*)malloc(width*sizeof(std::complex<float>));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) { 
      buffer[x] = src[y * width + x]; 
    }
    memcpy(bufferClone, buffer, width*sizeof(std::complex<float>));

    recursiveFastFourierTransformCPU(buffer, bufferClone, width, 1);

    for (int x = 0; x < width; x++) {
      imageBuffer[x * height + y] = buffer[x];
    }
  }

  for (int y = 0; y < height; y++) {
    memcpy(buffer, &imageBuffer[y * width], width*sizeof(std::complex<float>));
    memcpy(bufferClone, buffer, width*sizeof(std::complex<float>));

    recursiveFastFourierTransformCPU(buffer, bufferClone, width, 1);

    for (int x = 0; x < width; x++) {
      imageBuffer[y * width + x] = buffer[x];
    }
  }

  for (int x = 0; x < width * height; x++) {
    dst[x] = sqrtf((imageBuffer[x].real() * imageBuffer[x].real()) + (imageBuffer[x].imag() * imageBuffer[x].imag()));
  }
}