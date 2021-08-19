CXX = g++
CXXFLAGS  = -Xcompiler -Wall -Xcompiler -fopenmp -D USE_OPENMP -Xcompiler -std=c++03

mkfile_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LDFLAGS = -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ \
	-lhdf5 -lhdf5_cpp -I/home/alex/Downloads/boost_1_75_0 

NVCC = /usr/local/cuda/bin/nvcc
NVFLAGS = -O3 --expt-relaxed-constexpr --gpu-architecture sm_70 -lineinfo -allow-unsupported-compiler

cbet-gpu: def.cuh main.cu launch_ray_XZ.cu
	$(NVCC) $(NVFLAGS) $(CXXFLAGS) -o cbet-gpu main.cu launch_ray_XZ.cu $(LDFLAGS)

test: def.cuh main.cu launch_ray_XZ.cu
	$(NVCC) $(NVFLAGS) $(CXXFLAGS) -D PRINT -o cbet-gpu main.cu launch_ray_XZ.cu $(LDFLAGS)
	./cbet-gpu 10 > cbet_gpu_output
	cmp cbet_gpu_output truth_100

.PHONY: clean
clean:
	$(RM) cbet-gpu
