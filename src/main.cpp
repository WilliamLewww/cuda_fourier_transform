#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <stdio.h>

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize.h"

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
}

int main(int argn, char** argv) {
  int fourierWidth = 500, fourierHeight = 500, channels = 4;

  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, STBI_rgb_alpha);
  unsigned char* imageScaled = (unsigned char*)malloc(fourierWidth*fourierHeight*channels*sizeof(unsigned char));

  stbir_resize_uint8(image, width, height, 0, imageScaled, fourierWidth, fourierHeight, 0, channels);

  unsigned char* imageFourier = (unsigned char*)malloc(fourierWidth*fourierHeight*4*sizeof(unsigned char));
  fourierTransformWrapper(imageFourier, imageScaled, fourierWidth, fourierHeight, channels);

  stbi_write_png(argv[2], fourierWidth, fourierHeight, channels, imageFourier, fourierWidth*channels*sizeof(unsigned char));

  free(imageScaled);
  stbi_image_free(image);
  return 0;
}