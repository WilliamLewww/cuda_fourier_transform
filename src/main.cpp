#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <chrono>
#include <vector>
#include <string>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize.h"

extern "C" {
  void discreteFourierTransformWrapper(unsigned char* dst, unsigned char* src, int width, int height, int channels);
  void discreteFourierTransformBatchWrapper(unsigned char* dst, unsigned char* src, int width, int height, int depth, int channels);
}

void fourierTransformFile(std::string inputFile, std::string outputFile, int outputWidth, int outputHeight, int channels) {
  int width, height, comp;
  unsigned char* image = stbi_load(inputFile.c_str(), &width, &height, &comp, channels);
  unsigned char* imageScaled = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));

  stbir_resize_uint8(image, width, height, 0, imageScaled, outputWidth, outputHeight, 0, channels);

  unsigned char* imageFourier = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));

  printf("%s %s %dx%d", inputFile.c_str(), outputFile.c_str(), outputWidth, outputHeight);
  
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  discreteFourierTransformWrapper(imageFourier, imageScaled, outputWidth, outputHeight, channels);
  int64_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs", "kernel execution:", float(timeDifference) / 1000000.0);

  start = std::chrono::high_resolution_clock::now();
  stbi_write_png(outputFile.c_str(), outputWidth, outputHeight, channels, imageFourier, outputWidth*channels*sizeof(unsigned char));
  timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs\n", "file save:", float(timeDifference) / 1000000.0);

  free(imageFourier);
  free(imageScaled);
  stbi_image_free(image);
}

void fourierTransformDirectory(std::string inputDirectory, std::string outputDirectory, int outputWidth, int outputHeight) {
  int channels = 0;
  std::vector<std::string> fileList;

  DIR* directory;
  struct dirent* entry;

  if ((directory = opendir(inputDirectory.c_str())) != NULL) {
    while ((entry = readdir(directory)) != NULL) {
      std::string fileExtension = std::string(entry->d_name);
      if (fileExtension.size() > 3) {
        fileExtension = fileExtension.substr(fileExtension.size() - 4);

        if (fileExtension == ".jpg") { channels = 3; }
        if (fileExtension == ".png") { channels = 4; }

        fileList.push_back(entry->d_name);
      }
    }
    closedir(directory);
  }

  unsigned char* imageScaledArray = (unsigned char*)malloc(outputWidth*outputHeight*channels*fileList.size()*sizeof(unsigned char));

  for (int x = 0; x < fileList.size(); x++) {
    std::string inputFile = inputDirectory + std::string("/") + fileList[x];

    int width, height, comp;
    unsigned char* image = stbi_load(inputFile.c_str(), &width, &height, &comp, channels);    
    unsigned char* imageScaled = (unsigned char*)malloc(outputWidth*outputHeight*channels*sizeof(unsigned char));
    stbir_resize_uint8(image, width, height, 0, imageScaled, outputWidth, outputHeight, 0, channels);

    memcpy(imageScaledArray + (outputWidth * outputHeight * channels * x), imageScaled, outputWidth*outputHeight*channels*sizeof(unsigned char));

    free(image);
    free(imageScaled);
  }

  unsigned char* imageArrayFourier = (unsigned char*)malloc(outputWidth*outputHeight*channels*fileList.size()*sizeof(unsigned char));

  printf("%s %s %dx%d", inputDirectory.c_str(), outputDirectory.c_str(), outputWidth, outputHeight);

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  discreteFourierTransformBatchWrapper(imageArrayFourier, imageScaledArray, outputWidth, outputHeight, fileList.size(), channels);
  int64_t timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs", "kernel execution:", float(timeDifference) / 1000000.0);

  start = std::chrono::high_resolution_clock::now();
  for (int x = 0; x < fileList.size(); x++) {
    std::string outputFile = outputDirectory + std::string("/") + fileList[x];
    stbi_write_png(outputFile.c_str(), outputWidth, outputHeight, channels, &imageArrayFourier[outputWidth * outputHeight * channels * x], outputWidth*channels*sizeof(unsigned char));
  }
  timeDifference = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
  printf(" || %s %fs\n", "file save:", float(timeDifference) / 1000000.0);

  free(imageArrayFourier);
  free(imageScaledArray);
}

int main(int argn, char** argv) {
  if (argn == 3 || argn == 5) {
    int outputWidth = 500;
    int outputHeight = 500;
    if (argn == 5) {
      outputWidth = atoi(argv[3]);
      outputHeight = atoi(argv[4]);
    }

    int channels = 0;
    char* fileExtension = &argv[1][strlen(argv[1]) - 4];
    if (strcmp(fileExtension, ".jpg") == 0) { channels = 3; }
    if (strcmp(fileExtension, ".png") == 0) { channels = 4; }

    if (channels != 0) {
      fourierTransformFile(argv[1], argv[2], outputWidth, outputHeight, channels);
    }
    else {
      mkdir(argv[2], S_IRWXU | S_IRWXG | S_IRWXO);
      fourierTransformDirectory(argv[1], argv[2], outputWidth, outputHeight);
    }
  }

  return 0;
}