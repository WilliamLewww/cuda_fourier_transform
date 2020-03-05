#include <stdio.h>
#include <cmath>
#include <complex>
#include <cstring>

void circularShiftCPU(float* dst, float* src, int width, int height, int shiftX, int shiftY) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      dst[y * width + x] = src[(((y - shiftY) % height + height) % height) * width + (((x - shiftX) % width + width) % width)];
    }
  }
}

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
  float* image = (float*)malloc(width*height*sizeof(float));
  float* imageRotated = (float*)malloc(width*height*sizeof(float));

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
      image[y * width + x] = sqrtf((buffer[x].real() * buffer[x].real()) + (buffer[x].imag() * buffer[x].imag()));
    }
  }

  circularShiftCPU(imageRotated, image, width, height, width / 2, height / 2);

  float max = 0.0;
  for (int x = 0; x < width * height; x++) {
    if (imageRotated[x] > max) {
      max = imageRotated[x];
    }
  }

  float c = 255.0 / log(1 + fabs(max));
  for (int x = 0; x < width * height; x++) {
    imageRotated[x] = c * log(1 + fabs(imageRotated[x]));
  }

  memcpy(dst, imageRotated, width*height*sizeof(float));

  free(bufferClone);
  free(buffer);
  free(imageBuffer);
  free(imageRotated);
  free(image);
}