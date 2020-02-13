#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>

#include "stb_image.h"
#include "stb_image_write.h"

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
}

int main(int argn, char** argv) {
  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, STBI_rgb_alpha);
  unsigned char* imageFourier = (unsigned char*)malloc(width*height*4*sizeof(unsigned char));

  fourierTransformWrapper(imageFourier, image, width, height, 4);
  stbi_write_png(argv[2], width, height, 4, imageFourier, width*4*sizeof(unsigned char));

  free(imageFourier);
  stbi_image_free(image);
  return 0;
}