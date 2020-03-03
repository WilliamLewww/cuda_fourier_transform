#include <stdio.h>
#include <cmath>
#include <complex>

void recursiveFourierTransformCPU(std::complex<double> buf[], std::complex<double> out[], int n, int step) {
  if (step < n) {
    recursiveFourierTransformCPU(out, buf, n, step * 2);
    recursiveFourierTransformCPU(out + step, buf + step, n, step * 2);
 
    for (int i = 0; i < n; i += 2 * step) {
      std::complex<double> I(0, 1);

      std::complex<double> t = std::exp(-I * M_PI * double(i) / double(n)) * out[i + step];
      buf[i / 2]     = out[i] + t;
      buf[(i + n)/2] = out[i] - t;
    }
  }
}

void fastFourierTransformCPU(float* dst, float* src, int width, int height) {
  std::complex<double> buffer[] = {1, 1, 1, 1, 0, 0, 0, 0};
  std::complex<double> bufferClone[] = {1, 1, 1, 1, 0, 0, 0, 0};

  recursiveFourierTransformCPU(buffer, bufferClone, 8, 1);

  for (int x = 0; x < 8; x++) {
    printf("%lf %lf\n", buffer[x].real(), buffer[x].imag());
  }
}