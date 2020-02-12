#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
}

int main(int argn, char** argv) {
  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, STBI_rgb_alpha);
  unsigned char* imageFourier = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

  fourierTransformWrapper(imageFourier, image, width, height, 4);

  stbi_image_free(image);
  return 0;
}