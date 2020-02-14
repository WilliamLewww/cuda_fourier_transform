CUDA_PATH=/usr/local/cuda-10.1

CURRENT_PATH=$(shell pwd)

BIN_PATH=$(CURRENT_PATH)/bin
SRC_PATH=$(CURRENT_PATH)/src

CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc
NVPROF=$(CUDA_PATH)/bin/nvprof
NSIGHT_CLI=$(CUDA_PATH)/bin/nv-nsight-cu-cli
NVVP=$(CUDA_PATH)/bin/nvvp
GDB=gdb
CUDA_GDB=$(CUDA_PATH)/bin/cuda-gdb
MEMCHECK=$(CUDA_PATH)/bin/cuda-memcheck

CUDA_FLAGS=--gpu-architecture=sm_30
LINKER_ARGUMENTS=

EXEC=fourier_transform.out
EXEC_ARGS=res/window/3.png bin/output.png 500 500

all: clean $(EXEC) run

SRCS := $(wildcard src/*.cpp)
OBJS := $(SRCS:src/%.cpp=%.o)

CUDA_SRCS := fourier_transform.cu
CUDA_OBJS := $(CUDA_SRCS:%.cu=%.o)

$(EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(CUDA_FLAGS) $(BIN_PATH)/*.o -o $(BIN_PATH)/$(EXEC) $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cpp
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/$@ $(LINKER_ARGUMENTS)

%.o: $(SRC_PATH)/%.cu
	$(NVCC) $(CUDA_FLAGS) --device-c $^ -o $(BIN_PATH)/$@ $(LINKER_ARGUMENTS)

run:
	$(BIN_PATH)/$(EXEC) $(EXEC_ARGS)

profile:
	sudo $(NVPROF) $(BIN_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)/profile.log; cat $(BIN_PATH)/profile.log;

nvvp:
	sudo $(NVVP) $(CURRENT_PATH)/bin/$(EXEC) $(EXEC_ARGS) -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

gdb:
	gdb $(CURRENT_PATH)/bin/$(EXEC)

cuda-gdb:
	$(CUDA_GDB) $(BIN_PATH)/$(EXEC)

memory-check:
	$(MEMCHECK) $(BIN_PATH)/$(EXEC) $(EXEC_ARGS) 2>$(BIN_PATH)/memory-check.log; cat $(BIN_PATH)/memory-check.log;

clean:
	rm -rf $(BIN_PATH)/*
	mkdir -p bin