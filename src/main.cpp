#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <chrono>
#include <vector>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize.h"

extern "C" {
  void fourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
  void fourierTransformBatchWrapper(unsigned char* dst, unsigned char* src, int width, int height, int depth, int channels);
}

void fourierTransformFile(char* inputFile, char* outputFile, int outputWidth, int outputHeight, int channels) {
  int width, height, comp;
  unsigned char* image = stbi_load(inputFile, &width, &height, &comp, channels);
  unsigned char* imageScaled = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));

  stbir_resize_uint8(image, width, height, 0, imageScaled, outputWidth, outputHeight, 0, channels);

  unsigned char* imageFourier = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));

  printf("%s %s %dx%d", inputFile, outputFile, outputWidth, outputHeight);
  
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  fourierTransformWrapper(imageFourier, imageScaled, outputWidth, outputHeight, channels);
  int64_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs", "kernel execution:", float(timeDifference) / 1000000.0);

  start = std::chrono::high_resolution_clock::now();
  stbi_write_png(outputFile, outputWidth, outputHeight, channels, imageFourier, outputWidth*channels*sizeof(unsigned char));
  timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs\n", "file save:", float(timeDifference) / 1000000.0);

  free(imageFourier);
  free(imageScaled);
  stbi_image_free(image);
}

void fourierTransformDirectory(char* inputDirectory, char* outputDirectory, int outputWidth, int outputHeight, int channels) {
  std::vector<char*> fileList;

  DIR* directory;
  struct dirent* entry;

  if ((directory = opendir(inputDirectory)) != NULL) {
    while ((entry = readdir(directory)) != NULL) {
      char* fileExtension = &entry->d_name[strlen(entry->d_name) - 4];
      if (strcmp(fileExtension, ".jpg") == 0 || strcmp(fileExtension, ".png") == 0) {
        char* filename = (char*)malloc(strlen(entry->d_name) + 1);
        strcpy(filename, entry->d_name);
        fileList.push_back(filename);
      }
    }
    closedir(directory);
  }

  unsigned char* imageScaledArray = (unsigned char*)malloc(outputWidth*outputHeight*channels*fileList.size()*sizeof(unsigned char));

  for (int x = 0; x < fileList.size(); x++) {
    char* inputFile = (char*)malloc(strlen(inputDirectory) + strlen(fileList[x]) + 2);
    strcat(strcat(strcpy(inputFile, inputDirectory), "/"), fileList[x]);

    int width, height, comp;
    unsigned char* image = stbi_load(inputFile, &width, &height, &comp, channels);    
    unsigned char* imageScaled = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));
    stbir_resize_uint8(image, width, height, 0, imageScaled, outputWidth, outputHeight, 0, channels);

    memcpy(imageScaledArray + (outputWidth * outputHeight * channels * x), imageScaled, outputWidth*outputHeight*channels*sizeof(unsigned char));

    free(image);
    free(imageScaled);
    free(inputFile);
  }

  unsigned char* imageArrayFourier = (unsigned char*)malloc(outputWidth*outputHeight*channels*fileList.size()*sizeof(unsigned char));

  printf("%s %s %dx%d", inputDirectory, outputDirectory, outputWidth, outputHeight);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  fourierTransformBatchWrapper(imageArrayFourier, imageScaledArray, outputWidth, outputHeight, fileList.size(), channels);
  int64_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs", "kernel execution:", float(timeDifference) / 1000000.0);

  start = std::chrono::high_resolution_clock::now();
  for (int x = 0; x < fileList.size(); x++) {
    char* outputFile = (char*)malloc(strlen(outputDirectory) + strlen(fileList[x]) + 2);
    strcat(strcat(strcpy(outputFile, outputDirectory), "/"), fileList[x]);

    stbi_write_png(outputFile, outputWidth, outputHeight, channels, &imageArrayFourier[outputWidth * outputHeight * channels * x], outputWidth*channels*sizeof(unsigned char));
  
    free(outputFile);
  }
  timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs\n", "file save:", float(timeDifference) / 1000000.0);

  for (int x = 0; x < fileList.size(); x++) {
    delete fileList[x];
  }
  free(imageArrayFourier);
  free(imageScaledArray);
}

int main(int argn, char** argv) {
  if (argn == 5) {
    int channels = 0;

    char* fileExtension = &argv[1][strlen(argv[1]) - 4];
    if (strcmp(fileExtension, ".jpg") == 0) { channels = 3; }
    if (strcmp(fileExtension, ".png") == 0) { channels = 4; }

    if (channels != 0) {
      fourierTransformFile(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]), channels);
    }
    else {
      mkdir(argv[2], S_IRWXU | S_IRWXG | S_IRWXO);
      fourierTransformDirectory(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]), 4);
    }
  }

  return 0;
}