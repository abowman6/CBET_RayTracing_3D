#ifndef __DEF_H_
#define __DEF_H_

//#include <algorithm>
//#include <array>
#include <cmath>
//#include <vector>
//#include <tuple>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#include <H5Cpp.h>
#include <cuda_runtime.h>

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/cstdlib.hpp>
#include <boost/next_prior.hpp>
#include "multi_gpu.cuh"
#include "type.h"
#include "omega_beams.h"
using namespace std;


// Define constants

#define marked_const 1500 

// Deine the 2D cartesian extent of the grid in cm (1e-4 is 1 micron).
#define xyz_size 100
const static int nx = xyz_size;
const static int ny = xyz_size;
const static int nz = xyz_size;

#define nbeams 60 

#ifndef nGPUS
const static int nGPUs = 1;
#endif

const static int absorption = 1;
#define focal_length 0.1
/* Define Matrix Types*/
typedef boost::multi_array<dtype, 3> Array3D;
typedef Array3D::index Array3DIdx;
typedef boost::multi_array<dtype, 4> Array4D;
typedef Array4D::index Array4DIdx;
typedef boost::multi_array<int, 4> Array4I;
typedef Array4I::index Array4IIdx;

// >= 5k
const static int max_threads = 120000000;
const static int threads_per_block = 256;
/* Piecewise linear interpolation
   Use binary search to find the segment
   Ref: https://software.llnl.gov/yorick-doc/qref/qrfunc09.html
*/
dtype interp(const vector<dtype> y, const vector<dtype> x, const dtype xp);

__global__
void launch_ray_XYZ(int b, unsigned nindices,
                   dtype *dedendx, dtype *dedendy, dtype *dedendz,
                   dtype *edep, dtype *bbeam_norm,
                   dtype *myx_arr, dtype *myy_arr, 
                   int *marked, int *boxes, dtype *ray_coverage, int *counter);
#endif
