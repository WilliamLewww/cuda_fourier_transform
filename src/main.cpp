#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height);
}

int main(int argn, char** argv) {
  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, STBI_rgb_alpha);
  unsigned char* imageFourier = (unsigned char*)malloc(width * height * sizeof(unsigned char));

  fourierTransformWrapper(imageFourier, image, width, height);

  stbi_image_free(image);
  return 0;
}