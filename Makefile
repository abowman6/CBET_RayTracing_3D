CXX = g++
CXXFLAGS  = -Xcompiler -Wall -Xcompiler -fopenmp -D USE_OPENMP -Xcompiler -pg

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LDFLAGS = -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ \
	-lhdf5 -lhdf5_cpp

NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = --expt-relaxed-constexpr --gpu-architecture sm_70 -lineinfo -g -Xptxas -O3,-v -allow-unsupported-compiler

cbet-gpu: def.cuh main.cu launch_ray_XZ.cu multi_gpu.cpp
	$(NVCC) $(NVFLAGS) $(CXXFLAGS) -o cbet-gpu main.cu launch_ray_XZ.cu multi_gpu.cpp $(LDFLAGS)

double : def.cuh main.cu launch_ray_XZ.cu multi_gpu.cpp
	$(NVCC) $(NVFLAGS) $(CXXFLAGS) -D DOUBLE -o cbet-gpu main.cu launch_ray_XZ.cu multi_gpu.cpp $(LDFLAGS)
test: def.cuh main.cu launch_ray_XZ.cu
	$(NVCC) $(NVFLAGS) $(CXXFLAGS) -D DOUBLE -D PRINT -o cbet-gpu main.cu launch_ray_XZ.cu multi_gpu.cpp $(LDFLAGS)
	./cbet-gpu 10 > cbet_gpu_output
	cmp cbet_gpu_output truth_100

.PHONY: clean
clean:
	$(RM) cbet-gpu cbet_gpu_output
