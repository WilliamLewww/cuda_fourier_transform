#include <stdio.h>
#include <cmath>
#include <complex>

void recursiveFastFourierTransformCPU(std::complex<double>* buffer, std::complex<double>* out, int size, int stride) {
  if (stride < size) {
    recursiveFastFourierTransformCPU(out, buffer, size, stride * 2);
    recursiveFastFourierTransformCPU(out + stride, buffer + stride, size, stride * 2);
 
    for (int i = 0; i < size; i += 2 * stride) {
      std::complex<double> t = std::exp(-std::complex<double>(0, 1) * M_PI * double(i) / double(size)) * out[i + stride];
      buffer[i / 2] = out[i] + t;
      buffer[(i + size)/2] = out[i] - t;
    }
  }
}

void fastFourierTransformCPU(float* dst, float* src, int width, int height) {
  std::complex<double> buffer[] = {1, 1, 1, 1, 0, 0, 0, 0};
  std::complex<double> bufferClone[] = {1, 1, 1, 1, 0, 0, 0, 0};

  recursiveFastFourierTransformCPU(buffer, bufferClone, 8, 1);

  for (int x = 0; x < 8; x++) {
    printf("%f\n", sqrt((buffer[x].real() * buffer[x].real()) + (buffer[x].imag() * buffer[x].imag())));
  }
}