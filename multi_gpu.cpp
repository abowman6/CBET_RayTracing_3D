#include "multi_gpu.cuh"

bool safeGPUAlloc(void **dst, size_t size, int GPUIndex) {
    // if (!usable[GPUIndex]) {
    //     cout << "Assigning memory to unusable GPU" << endl;
    // }
    cudaSetDevice(GPUIndex);
    size_t free;
    size_t total;
    // get the amount of memory
    cudaError_t e = cudaMemGetInfo(&free, &total);
    if (e != cudaSuccess) {
        cout << "Error encountered during cudaMemGetInfo: " << cudaGetErrorString(e) << endl;
        return false;
    }
    // make sure we are not allocating memory on a GPU out of memory
    if (free >= size) {
        e = cudaMalloc(dst, size);
        if (e == cudaSuccess) { 
            return true;
        } else {
            cout << "Error encountered during cudaMalloc: " << cudaGetErrorString(e) << endl;
        }
    } else {
        cout << "GPU: " << GPUIndex << " is out of memory" << endl;
    }
    return false;
}

// int initializeOnGPU(void **dst, size_t size) {
//     int ret = -1;
//     int temp = 0;
//     cudaGetDevice(&temp);
//     for (int i = 0; i < numGPUs; ++i) {
// 	    if (!usable[i]) continue; // do not assign memory as other GPUs can't access it
//         if (safeGPUAlloc(dst, size, i)) {
//             return i;
//         }
//     }
//     cudaSetDevice(temp);
//     return ret;
// }

bool moveToAndFromGPU(void *dst, void *src, size_t size, int GPUIndex) {
    if (GPUIndex == -1) {
        cout << "Attempting to move data that has not been assigned a GPU" << endl;
        return false;
    }
    cudaError_t e = cudaSuccess;
    int temp = 0;
    cudaGetDevice(&temp);
    cudaSetDevice(GPUIndex);
    e = cudaMemcpy(dst, src, size, cudaMemcpyDefault);
    if (e != cudaSuccess) {
        cout << "Error encountered during cudaMemcpy: " << cudaGetErrorString(e) << endl;
    }
    cudaSetDevice(temp);
    return e == cudaSuccess;
}
