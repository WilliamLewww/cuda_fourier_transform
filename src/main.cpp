#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argn, char** argv) {
  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, STBI_rgb_alpha);

  stbi_image_free(image);
  return 0;
}