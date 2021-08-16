#include <cuda_runtime.h>
#include <iostream>

using namespace std;

extern bool safeGPUAlloc(void **dst, size_t size, int GPUIndex);
extern bool moveToAndFromGPU(void *dst, void *src, size_t size, int GPUIndex);

