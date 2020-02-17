#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <stdio.h>
#include <chrono>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize.h"

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
}

int main(int argn, char** argv) {
  int fourierWidth = atoi(argv[3]); 
  int fourierHeight = atoi(argv[4]);
  int channels;

  char* fileExtension = &argv[1][strlen(argv[1]) - 4];
  if (strcmp(fileExtension, ".jpg") == 0) { channels = 3; }
  if (strcmp(fileExtension, ".png") == 0) { channels = 4; }

  int width, height, comp;
  unsigned char* image = stbi_load(argv[1], &width, &height, &comp, channels);
  unsigned char* imageScaled = (unsigned char*)malloc(fourierWidth*fourierHeight*channels*sizeof(unsigned char));

  stbir_resize_uint8(image, width, height, 0, imageScaled, fourierWidth, fourierHeight, 0, channels);

  unsigned char* imageFourier = (unsigned char*)malloc(fourierWidth*fourierHeight*channels*sizeof(unsigned char));

  printf("%s %s %sx%s", argv[1], argv[2], argv[3], argv[4]);
  
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  fourierTransformWrapper(imageFourier, imageScaled, fourierWidth, fourierHeight, channels);
  int64_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs", "kernel execution:", float(timeDifference) / 1000000.0);

  start = std::chrono::high_resolution_clock::now();
  stbi_write_png(argv[2], fourierWidth, fourierHeight, channels, imageFourier, fourierWidth*channels*sizeof(unsigned char));
  timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs\n", "file save:", float(timeDifference) / 1000000.0);

  free(imageFourier);
  free(imageScaled);
  stbi_image_free(image);
  return 0;
}