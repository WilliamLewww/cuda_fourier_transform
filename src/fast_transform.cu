#include <stdio.h>
#include <cmath>
#include <complex>

void recursiveFastFourierTransformCPU(std::complex<double>* bufferCombine, std::complex<double>* bufferSplit, int size, int stride) {
  if (stride < size) {
    recursiveFastFourierTransformCPU(bufferSplit, bufferCombine, size, stride * 2);
    recursiveFastFourierTransformCPU(bufferSplit + stride, bufferCombine + stride, size, stride * 2);
 
    for (int i = 0; i < size; i += 2 * stride) {
      std::complex<double> t = std::exp(-std::complex<double>(0, 1) * M_PI * double(i) / double(size)) * bufferSplit[i + stride];
      bufferCombine[i / 2] = bufferSplit[i] + t;
      bufferCombine[(i + size)/2] = bufferSplit[i] - t;
    }
  }
}

void fastFourierTransformCPU(float* dst, float* src, int width, int height) {
  // std::complex<double> buffer[] = {1, 1, 1, 1, 0, 0, 0, 0};
  // std::complex<double> bufferClone[] = {1, 1, 1, 1, 0, 0, 0, 0};

  std::complex<double>* buffer = (std::complex<double>*)malloc(width*sizeof(std::complex<double>));
  for (int x = 0; x < width; x++) { buffer[x] = src[x]; }

  std::complex<double>* bufferClone = (std::complex<double>*)malloc(width*sizeof(std::complex<double>));
  memcpy(bufferClone, buffer, width*sizeof(std::complex<double>));

  recursiveFastFourierTransformCPU(buffer, bufferClone, width, 1);

  for (int x = 0; x < width; x++) {
    printf("%f\n", sqrtf((buffer[x].real() * buffer[x].real()) + (buffer[x].imag() * buffer[x].imag())));
  }
}