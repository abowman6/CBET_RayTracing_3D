#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <omp.h>
#include <H5Cpp.h>
#include <cuda_runtime.h>

#include "def.cuh"
#include "omega_beams.h"

int numGPUs = 1;
bool *usable;

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t b;

struct threadData {
    int gIndex;
    double **eden;
    double **edep;
    double **dedendx;
    double **dedendy;
    double **dedendz;
    double **etemp;
    double *myx;
    double *myy;
    double *myz;
    double *myvx;
    double *myvy;
    double *myvz;
    int *thisx;
    int *thisy;
    int *thisz;
    double *uray;
    double *uray0;
    int *update;
    int *counter;
    int *time_passed;
    int i;
    int j;
    int k;
    double *x;
};

struct kerRet {
    double *edep_chunk;
    int i;
    int j;
    int k;
};

void *thread_stall(void *d) {
    return NULL;
}

void addIn(double *edep, double *edep_chunk) {
   
    for (int xx = 0; xx < chunk_size+2; ++xx) {
        for (int yy = 0; yy < chunk_size+2; ++yy) {
            for (int zz = 0; zz < chunk_size+2; ++zz) {
                int index = xx*(chunk_size+2)*(chunk_size+2) + yy*(chunk_size+2) + zz;
                edep[index] = edep_chunk[index];                
            }
        }
    }
}

bool safeGPUAlloc(void **dst, size_t size, int GPUIndex) {
    if (!usable[GPUIndex]) {
        cout << "Assigning memory to unusable GPU" << endl;
    }
    cudaSetDevice(GPUIndex);
    size_t free;
    size_t total;
    // get the amount of memory
    cudaError_t e = cudaMemGetInfo(&free, &total);
    if (e != cudaSuccess) {
        cout << "Error encountered during cudaMemGetInfo: " << cudaGetErrorString(e) << endl;
        return false;
    }
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

int initializeOnGPU(void **dst, size_t size) {
    int ret = -1;
    int temp = 0;
    cudaGetDevice(&temp);
    for (int i = 0; i < numGPUs; ++i) {
	    if (!usable[i]) continue; // do not assign memory as other GPUs can't access it
        if (safeGPUAlloc(dst, size, i)) {
            return i;
        }
    }
    cudaSetDevice(temp);
    return ret;
}

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
void *runKer(void *data) {
    dim3 nblocks(nbeams/2, threads_per_beam, 1);
    struct threadData *d = (struct threadData *)data;
    int t = d->gIndex;
    double *devdedendx, *devdedendy, *devdedendz, *deden, *detemp, *dedep;
    int *counter = NULL;
    int zero = 0;
    safeGPUAlloc((void **)&devdedendx, sizeof(double)*c3, t);
    safeGPUAlloc((void **)&devdedendy, sizeof(double)*c3, t);
    safeGPUAlloc((void **)&devdedendz, sizeof(double)*c3, t);
    safeGPUAlloc((void **)&deden, sizeof(double)*c3, t);
    // safeGPUAlloc((void **)&detemp, sizeof(double)*c3, t);
    safeGPUAlloc((void **)&dedep, sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), t);
    safeGPUAlloc((void **)&counter, sizeof(int), t);
    int currThread = t;
    int curr = 0;
 
    for (int tt = 0;; ++tt) {
        for (int i = 0; i < nx; i+=chunk_size) {
            for (int j = 0; j < ny; j+=chunk_size) {
                for (int k = 0; k < nz; k+=chunk_size) {
                    int index = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                    if (index%2 == currThread) continue;
                    moveToAndFromGPU(devdedendx, d->dedendx[index], sizeof(double)*c3, t);
                    moveToAndFromGPU(devdedendy, d->dedendy[index], sizeof(double)*c3, t);
                    moveToAndFromGPU(devdedendz, d->dedendz[index], sizeof(double)*c3, t);
                    moveToAndFromGPU(deden, d->eden[index], sizeof(double)*c3, t);
                    // moveToAndFromGPU(detemp, d->etemp[index], sizeof(double)*c3, t);
                    moveToAndFromGPU(dedep, d->edep[index], sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), t);
                    moveToAndFromGPU(counter, &zero, sizeof(int), t);
 
                    cudaSetDevice(t);

                    launch_ray_XYZ<<<nblocks, threads_per_block>>>(t, nindices, NULL, NULL, NULL, devdedendx, devdedendy,
                    devdedendz, dedep, deden, detemp, d->uray, d->myx, d->myy, d->myz, d->myvx, d->myvy, d->myvz, d->uray0, d->thisx, d->thisy, d->thisz,
                    i, j, k, d->time_passed, d->update, counter);
                    
                    cudaDeviceSynchronize(); 
                    //cout << "Thread " << t << " completed chunk " << index << endl;
                    moveToAndFromGPU(d->edep[(i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+k/chunk_size], dedep, sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), t);
                    cudaMemcpy(&curr, counter, sizeof(int), cudaMemcpyDefault);
                    pthread_mutex_lock(&m);
                    *d->counter += curr;
                    pthread_mutex_unlock(&m);
                    //int curr = 0;
                }
            }
        }
        currThread = 1 - currThread;
        pthread_barrier_wait(&b);
        if (*d->counter == nbeams*nrays) {
            break;
        }
    }
    return NULL;
}

void print1d(ostream& f, vector<double> array) {
    std::cout.precision(8);
    for (vector<double>::size_type i = 0; i < array.size(); i++ )
        f << array[i] << ' ';
    f << std::endl;
}

void print2d(ostream& f, double **array, int w, int h) {
    // std::cout.precision(8);
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            f << array[i][j];
            if (j != h-1) {
                f << ",";
            }
        }
        f << endl;
    }
}

void print(std::ostream& os, const double& x)
{
  os << x;
}

template <typename Array>
void print(std::ostream& os, const Array& A)
{
  //std::cout.precision(16);
  typename Array::const_iterator i;
  os << "[";
  for (i = A.begin(); i != A.end(); i++) {
    print(os, *i);
    if (boost::next(i) != A.end())
      os << ',';
  }
  os << "]" << endl;
}

// linear space
vector<double> span(double minimum, double maximum, unsigned len) {
    double step = (maximum - minimum) / (len - 1), curr = minimum;
    vector<double> ret(len);
    for (unsigned i = 0; i < len; ++i) {
        ret[i] = curr;
        curr += step;
    }
    return ret;
}

double interp(const vector<double> y, const vector<double> x, const double xp)
{
    unsigned low, high, mid;
    assert(x.size() == y.size());

    if (x.front() <= x.back()) {
        // x monotonically increase
        if (xp <= x.front())
            return y.front();
        else if (xp >= x.back())
            return y.back();

        low = 0;
        high = x.size() - 1;
        mid = (low + high) >> 1;
        while (low < high - 1) {
            if (x[mid] >= xp)
                high = mid;
            else
                low = mid;
            mid = (low + high) >> 1;
        }

        assert((xp >= x[mid]) && (xp <= x[mid + 1]));
        return y[mid] +
            (y[mid + 1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    } else {
        // x monotonically decrease
        if (xp >= x.front())
            return y.front();
        else if (xp <= x.back())
            return y.back();

        low = 0;
        high = x.size() - 1;
        mid = (low + high) >> 1;
        while (low < high - 1) {
            if (x[mid] <= xp)
                low = mid;
            else
                high = mid;
            mid = (low + high) >> 1;
        }

        assert((xp <= x[mid]) && (xp >= x[mid + 1]));
        return y[mid] +
            (y[mid +1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    }
}

/*
   Save multi_array to hdf5 file
 */
int save2Hdf5(Array3D& x, Array3D& y, Array3D& z, Array3D& edepavg)
{
    const H5std_string  FILE_NAME("edep.hdf5");

    try
        {
            /*
             * Turn off the auto-printing when failure occurs so that we can
             * handle the errors appropriately
             */
            H5::Exception::dontPrint();
            /*
             * Create a new file using H5F_ACC_TRUNC access,
             * default file creation properties, and default file
             * access properties.
             */
            H5::H5File file(FILE_NAME, H5F_ACC_TRUNC);
            /*
             * Define the size of the array and create the data space for fixed
             * size dataset.
             */
            hsize_t     dimsf[3];              // dataset dimensions
            dimsf[0] = nx;
            dimsf[1] = ny;
            dimsf[2] = nz;
            H5::DataSpace dataspace(3, dimsf);
            /*
             * Define datatype for the data in the file.
             * We will store little endian INT numbers.
             */
            H5::IntType datatype(H5::PredType::NATIVE_DOUBLE);
            datatype.setOrder(H5T_ORDER_LE);
            /*
             * Create a new dataset within the file using defined dataspace and
             * datatype and default dataset creation properties.
             */
            H5::DataSet dataset = file.createDataSet("/Coordinate_x", datatype, dataspace);
            dataset.write(x.data(), H5::PredType::NATIVE_DOUBLE);
            dataset = file.createDataSet("/Coordinate_y", datatype, dataspace);
            dataset.write(y.data(), H5::PredType::NATIVE_DOUBLE);
            dataset = file.createDataSet("/Coordinate_z", datatype, dataspace);
            dataset.write(z.data(), H5::PredType::NATIVE_DOUBLE);

            dataset = file.createDataSet("/Edepavg", datatype, dataspace);
            /*
             * Write the data to the dataset using default memory space, file
             * space, and transfer properties.
             */
            dataset.write(edepavg.data(), H5::PredType::NATIVE_DOUBLE);
        }  // end of try block
    // catch failure caused by the H5File operations
    catch(H5::Exception error )
        {
            error.printErrorStack();
            return -1;
        }
    return 0;  // successfully terminated
}

void rayTracing(vector<double> x, vector<double> y, vector<double> z,
    double **eden, double **edep, double **wpe, double **etemp, Array3D& edep2)
{
    // Array3D dedendx(boost::extents[nx][nz][nz]),
    //     dedendy(boost::extents[nx][nz][nz]),
    //     dedendz(boost::extents[nx][nz][nz]);

    // double **dedendx, **dedendy, **dedendz;

   // sz: depends on eden, which may need to reorder

    vector<double> phase_r = span(0.0, 0.1, 2001);
    vector<double> pow_r(2001);
    for (unsigned i = 0; i < 2001; ++i) {
        pow_r[i] = exp(-1*pow(pow((phase_r[i]/sigma),2), (5.0/2.0)));
    }

    cudaError_t e = cudaGetDeviceCount(&numGPUs);
    if (e != 0) {
        cout << cudaGetErrorString(e) << endl;
        return;
    }
    usable = new bool[numGPUs];
    int numUsableGPUs = numGPUs;

    // for simplicity allow all devices to access each other's memory
    // some GPUs wont have access the other's memory, we need to make
    // sure we are not assigning memory to them.
    if (numGPUs != 1) {
        for (int i = 0; i < numGPUs; ++i) {
            cudaSetDevice(i);
            bool canUse = false;
            for (int j = 0; j < numGPUs; ++j) {
                if (i == j) continue;
                e = cudaDeviceEnablePeerAccess(j, 0);
                canUse |= (e == 0);
            }
            if (canUse) {
                usable[i] = true;
            } else {
                usable[i] = false;
                numUsableGPUs--;
            }
        }
    } else {
        usable[0] = true;
    }

    // ab: arrays needed for the simulationj
    double *devx,*devy,*devz,*dwpe,/***devdedendx,**devdedendy,**devdedendz,**dedep,*/
        /***deden,**detemp,*/*dbeam_norm,*dpow_r,*dphase_r;
    //double *devx2,*devy2,*devz2,*dwpe2,*devdedendx2,*devdedendy2,*devdedendz2,*dedep2,
    //   *deden2,*detemp2;
    // ab: arrays to keep track of ray states
    double *myx, *myy, *myz, *myvx, *myvy, *myvz, *uray_arr, *uray_init;
    int *thisx_0, *thisy_0, *thisz_0, *time_passed, *update;
    int *counter;

    safeGPUAlloc((void **)&counter, sizeof(int), 0);

    // ab: assuming 1 or 2 GPUs for now
    safeGPUAlloc((void **)&myx, sizeof(double)*nrays*nbeams, 0);
    safeGPUAlloc((void **)&myy, sizeof(double)*nrays*nbeams, 0);
    safeGPUAlloc((void **)&myz, sizeof(double)*nrays*nbeams, 0);
    safeGPUAlloc((void **)&myvx, sizeof(double)*nrays*nbeams, 0);
    safeGPUAlloc((void **)&myvy, sizeof(double)*nrays*nbeams, 0);
    safeGPUAlloc((void **)&dbeam_norm, sizeof(double)*180, 0);
    safeGPUAlloc((void **)&dpow_r, sizeof(double)*2001, 0);
    safeGPUAlloc((void **)&dphase_r, sizeof(double)*2001, 0);
    safeGPUAlloc((void **)&myvz, sizeof(double)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&thisx_0, sizeof(int)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&thisy_0, sizeof(int)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&thisz_0, sizeof(int)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&time_passed, sizeof(int)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&uray_arr, sizeof(double)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&uray_init, sizeof(double)*nrays*nbeams, numUsableGPUs-1);
    safeGPUAlloc((void **)&update, sizeof(int)*nrays*nbeams, numUsableGPUs-1);

    moveToAndFromGPU(dbeam_norm, &(beam_norm[0][0]), sizeof(double)*180, 0);
    moveToAndFromGPU(dpow_r, &(pow_r[0]), sizeof(double)*2001, 0);
    moveToAndFromGPU(dphase_r, &(phase_r[0]), sizeof(double)*2001, 0);
/*
    double **devdedendx = (double **)malloc(sizeof(double *)*2);
    double **devdedendy = (double **)malloc(sizeof(double *)*2);
    double **devdedendz = (double **)malloc(sizeof(double *)*2);
    double **deden = (double **)malloc(sizeof(double *)*2);
    double **detemp = (double **)malloc(sizeof(double *)*2);
    double **dedep = (double **)malloc(sizeof(double *)*2);
*/

    /*double *devdedendx1 = NULL;
    double *devdedendx2 = NULL;
    double *devdedendy1 = NULL;
    double *devdedendy2 = NULL;
    double *devdedendz1 = NULL;
    double *devdedendz2 = NULL;
    double *deden1 = NULL;
    double *deden2 = NULL;
    double *detemp1 = NULL;
    double *detemp2 = NULL;
    double *dedep1 = NULL;
    double *dedep2 = NULL;*/
     
    //safeGPUAlloc((void **)&devx, sizeof(double)*nx, 0);
    //safeGPUAlloc((void **)&devy, sizeof(double)*ny, 0);
    //safeGPUAlloc((void **)&devz, sizeof(double)*nz, 0);
    
/*
    safeGPUAlloc((void **)&(devdedendx[0]), sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    safeGPUAlloc((void **)&(devdedendy[0]), sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    safeGPUAlloc((void **)&(devdedendz[0]), sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    safeGPUAlloc((void **)&(deden[0]), sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    safeGPUAlloc((void **)&(detemp[0]), sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    safeGPUAlloc((void **)&(dedep[0]), sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), 0);

    safeGPUAlloc((void **)&(devdedendx[1]), sizeof(double)*chunk_size*chunk_size*chunk_size, 1);
    safeGPUAlloc((void **)&(devdedendy[1]), sizeof(double)*chunk_size*chunk_size*chunk_size, 1);
    safeGPUAlloc((void **)&(devdedendz[1]), sizeof(double)*chunk_size*chunk_size*chunk_size, 1);
    safeGPUAlloc((void **)&(deden[1]), sizeof(double)*chunk_size*chunk_size*chunk_size, 1);
    safeGPUAlloc((void **)&(detemp[1]), sizeof(double)*chunk_size*chunk_size*chunk_size, 1);
    safeGPUAlloc((void **)&(dedep[1]), sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), 1);
*/
/*
    safegpualloc((void **)&devdedendx1, sizeof(double)*c3, 0);
    safegpualloc((void **)&devdedendx2, sizeof(double)*c3, 1);
    safegpualloc((void **)&devdedendy1, sizeof(double)*c3, 0);
    safegpualloc((void **)&devdedendy2, sizeof(double)*c3, 1);
    safegpualloc((void **)&devdedendz1, sizeof(double)*c3, 0);
    safegpualloc((void **)&devdedendz2, sizeof(double)*c3, 1);
    safegpualloc((void **)&deden1, sizeof(double)*c3, 0);
    safegpualloc((void **)&deden2, sizeof(double)*c3, 1);
    safegpualloc((void **)&detemp1, sizeof(double)*c3, 0);
    safegpualloc((void **)&detemp2, sizeof(double)*c3, 1);
    safegpualloc((void **)&dedep1, sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), 0);
    safeGPUAlloc((void **)&dedep2, sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), 1);
*/

    // ab: will be needed in the future
/*
    if (numUsableGPUs > 1) {
        safeGPUAlloc((void **)&devx2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&devy2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&devz2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&devdedendx2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&devdedendy2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&devdedendz2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&dwpe2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&deden2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&detemp2, sizeof(double)*chunk_size*chunk_size*chunk_size, numUsableGPUs-1);
        safeGPUAlloc((void **)&dedep2, sizeof(double)*(chunk_size+2)*(chunk_size+2)*(chunk_size+2), numUsableGPUs-1);
    }
*/

    //moveToAndFromGPU(devx, x, sizeof(double)*nx, 0);
    //moveToAndFromGPU(devy, y, sizeof(double)*ny, 0);
    //moveToAndFromGPU(devz, z, sizeof(double)*nz, 0);

    // ab: this one is quick, not worth splitting
    // cudaSetDevice(0); 
  
    dim3 nblocks(nbeams, threads_per_beam, 1);
    calculate_myxyz<<<nblocks, threads_per_block>>>(nindices, dbeam_norm, dpow_r, dphase_r, uray_arr, uray_init, time_passed, thisx_0, 
        thisy_0, thisz_0, devx, devy, devz, counter, myx, myy, myz, uray_mult, focal_length);


    cudaDeviceSynchronize();
    wpe = (double **)malloc(sizeof(double *)*nchunks);
    for (int i = 0; i < nchunks; ++i) {
        wpe[i] = new double[c3];
    }
    safeGPUAlloc((void **)&dwpe, sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
    cudaSetDevice(0);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                int idx1 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx2 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                tmp = interp(z, y, tmp);
                wpe[idx1][idx2] = sqrt((pow(omega, 2) - pow(sqrt(tmp*1e6*pow(ec,2)/((double)me*e0)), 2)) /
                        pow(c, 2.0));
            }
        }
    }
    for (int i = 0; i < nx; i+=chunk_size) {
        for (int j = 0; j < ny; j+=chunk_size) {
            for (int k = 0; k < nz; k+=chunk_size) {
                int index = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                moveToAndFromGPU(dwpe, wpe[index], sizeof(double)*chunk_size*chunk_size*chunk_size, 0);
                initial_launch<<<nblocks, threads_per_block>>>(nindices, dwpe,
                    dbeam_norm, uray_arr, thisx_0, thisy_0, thisz_0, myvx, myvy, myvz, time_passed, i, j, k);
                cudaDeviceSynchronize();
                //if (i == 50 && j == 50 && k == 0) {
                  //  double *temp = new double[c3];
                  //  cudaMemcpy(temp, dwpe, sizeof(double)*c3, cudaMemcpyDefault);
                    //printf("%lf %lf\n", temp[19*50*50+17*50+20], wpe[index][19*50*50+17*50+20]);
                //}
            }
        }
    }  
    for (int i = 0; i < nchunks; ++i) {
        delete [] wpe[i]; 
    }
    free(wpe);
    eden = (double **)malloc(sizeof(double *)*nchunks);
    // double **etemp = (double **)malloc(sizeof(double *)*nchunks);
    edep = (double **)malloc(sizeof(double *)*nchunks);
    for (int i = 0; i < nchunks; ++i) {
        eden[i] = new double[c3];
        // etemp[i] = new double[c3];
        edep[i] = new double[(int)pow((chunk_size+2), 3)];
    }
    /* Calculate the electron density using a function of x and z, as desired. */
    /* sz: Initialize eden and machnum using xz, no dependence among iterations */
#if 0
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                int idx1 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx2 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                eden[idx1][idx2] = interp(z, y, tmp);
                //printf("%lf\n", interp(z, y, tmp));
            }
        }
    }
    double **dedendx = (double **)malloc(sizeof(double *)*nchunks);
    double **dedendy = (double **)malloc(sizeof(double *)*nchunks);
    double **dedendz = (double **)malloc(sizeof(double *)*nchunks);
    for (int i = 0; i < nchunks; ++i) {
        dedendx[i] = new double[c3];
        dedendy[i] = new double[c3];
        dedendz[i] = new double[c3];
        for (int j = 0; j < c3; ++j) {
            dedendx[i][j] = 0;
            dedendy[i][j] = 0;
            dedendz[i][j] = 0;
        }
    }
    /* Calculate the gradients of electron density w.r.t. x and z */
    // Central differences
    /* sz: Initialize dedendx/z using eden and xz, no dependence among iterations
       The last row/col needs to be treated differently
     */
#if 0
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
#endif
    for (unsigned i = 1; i < nx - 1; ++i){
        for (unsigned j = 1; j < ny - 1; ++j){
            for (unsigned k = 1; k < nz - 1; ++k){
                int idx1 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx2 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                int idx3 = ((i+1)/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx4 = ((i+1)%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                int idx5 = ((i-1)/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx6 = ((i-1)%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                int idx7 = (i/chunk_size)*nchunksy*nchunksz+((j+1)/chunk_size)*nchunksz+(k/chunk_size);
                int idx8 = (i%chunk_size)*chunk_size*chunk_size+((j+1)%chunk_size)*chunk_size+k%chunk_size;
                int idx9 = (i/chunk_size)*nchunksy*nchunksz+((j-1)/chunk_size)*nchunksz+(k/chunk_size);
                int idx10 = (i%chunk_size)*chunk_size*chunk_size+((j-1)%chunk_size)*chunk_size+k%chunk_size;
                int idx11 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+((k+1)/chunk_size);
                int idx12 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+(k+1)%chunk_size;
                int idx13 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+((k-1)/chunk_size);
                int idx14 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+(k-1)%chunk_size;
                //pow(c, 2) / (2.0 * ncrit) * dt * 
                dedendx[idx1][idx2] = pow(c, 2) / (2.0 * ncrit) * dt * 0.5 * (eden[idx3][idx4] - eden[idx5][idx6]) / xres;
                dedendy[idx1][idx2] = pow(c, 2) / (2.0 * ncrit) * dt * 0.5 * (eden[idx7][idx8] - eden[idx9][idx10]) / yres;
                dedendz[idx1][idx2] = pow(c, 2) / (2.0 * ncrit) * dt * 0.5 * (eden[idx11][idx12] - eden[idx13][idx14]) / zres;
                //int ii = idx1;
                //int jj = idx2;
                //double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                // dedendx[i][j][k] = (0.5 * (eden[i+1][j][k] + eden[i][j][k]) -
                //                     0.5 * (eden[i][j][k] + eden[i-1][j][k])) / xres;
                // dedendy[i][j][k] = (0.5 * (eden[i][j+1][k] + eden[i][j][k]) -
                //                     0.5 * (eden[i][j][k] + eden[i][j-1][k])) / yres;
                // dedendz[i][j][k] = (0.5 * (eden[i][j][k+1] + eden[i][j][k]) -
                //                     0.5 * (eden[i][j][k] + eden[i][j][k-1])) / zres;
            }
        }
    }
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j){
            // ab: surely there is a better way to do this
            int idx1 = (i/chunk_size)*nchunksz+(j/chunk_size);
            int idx2 = (i%chunk_size)*chunk_size+j%chunk_size;
            int idx3 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size);
            int idx4 = (i%chunk_size)*chunk_size*chunk_size+j%chunk_size;
            int idx5 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz;
            int idx6 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size;
            int idx7 = ((nx-1)/chunk_size)*nchunksy*nchunksz+(i/chunk_size)*nchunksz+(j/chunk_size);
            int idx8 = ((nx-1)%chunk_size)*chunk_size*chunk_size+(i%chunk_size)*chunk_size+j%chunk_size;
            int idx9 = (i/chunk_size)*nchunksy*nchunksz+((ny-1)/chunk_size)*nchunksz+(j/chunk_size);
            int idx10 = (i%chunk_size)*chunk_size*chunk_size+((ny-1)%chunk_size)*chunk_size+j%chunk_size;
            int idx11 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+((nz-1)/chunk_size);
            int idx12 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+(nz-1)%chunk_size;
            // ab: it just keeps going
            int idx13 = chunk_size*chunk_size+(i%chunk_size)*chunk_size+j%chunk_size;
            int idx14 = chunk_size+(i%chunk_size)*chunk_size*chunk_size+j%chunk_size;
            int idx15 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+1;
            int idx16 = ((nx-2)/chunk_size)*nchunksy*nchunksz+(i/chunk_size)*nchunksz+(j/chunk_size);
            int idx17 = ((nx-2)%chunk_size)*chunk_size*chunk_size+(i%chunk_size)*chunk_size+j%chunk_size;
            int idx18 = (i/chunk_size)*nchunksy*nchunksz+((ny-2)/chunk_size)*nchunksz+(j/chunk_size);
            int idx19 = (i%chunk_size)*chunk_size*chunk_size+((ny-2)%chunk_size)*chunk_size+j%chunk_size;
            int idx20 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+((nz-2)/chunk_size);
            int idx21 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+(nz-2)%chunk_size;
            dedendx[idx1][idx2] = dedendx[idx1][idx13];
            dedendy[idx3][idx4] = dedendy[idx3][idx14];
            dedendz[idx5][idx6] = dedendz[idx5][idx15];
            dedendx[idx7][idx8] = dedendx[idx16][idx17];
            dedendy[idx9][idx10] = dedendy[idx18][idx19];
            dedendz[idx11][idx12] = dedendz[idx20][idx21];
            
        }
    }

    //printf("%lf\n", dedendy[0][0]);
    //printf("%lf\n", dedendy[0][10]);


#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i){
        for (unsigned j = 0; j < ny; ++j){
            for (unsigned k = 0; k < nz; ++k){
                int idx1 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx2 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                double et = 5.2e-5 * 10.0 / pow(interp(x, y, tmp), 1.5);
                double nuei = (1e6 * eden[idx1][idx2] * pow(ec, 2)/me) * et;
                eden[idx1][idx2] = nuei * eden[idx1][idx2]/ncrit * dt;
            }
        }
    }
    for (int i = 0; i < nchunks; ++i) {
        for (int j = 0; j < (chunk_size+2)*(chunk_size+2)*(chunk_size+2); ++j) {
            edep[i][j] = 0;
        }
    }
 
    /*
    double *temp = new double[nbeams*nrays];
    cudaMemcpy(temp, myvx, sizeof(double)*nbeams*nrays, cudaMemcpyDefault);
    for (int i = 0; i < nbeams*nrays; ++i) {
        if (temp[i] == 0) continue;
        //printf("%lf\n", temp[i]);
    }
    */
   /*
    cudaFree(myx);
    cudaFree(myy);
    cudaFree(myz);
    cudaFree(myvx);
    cudaFree(myvy);
    cudaFree(myvz);
    cudaFree(thisx_0);
    cudaFree(thisy_0);
    cudaFree(thisz_0);
    cudaFree(time_passed);
    cudaFree(update);
    cudaFree(uray);
    cudaFree(uray_init);
    */
    /*
    double **myxa = new double *[2];
    double **myya = new double *[2];
    double **myza = new double *[2];
    double **myvxa = new double *[2];
    double **myvya = new double *[2];
    double **myvza = new double *[2];
    int **thisx_0a = new int *[2];
    int **thisy_0a = new int *[2];
    int **thisz_0a = new int *[2];
    int **time_passeda = new int *[2];
    int **updatea = new int *[2];
    double **uraya = new double *[2];
    double **uray_inita = new double *[2];

    //printf("%d %d\n", nbeams*nrays, nbeams*nrays/2);

    for (int i = 0; i < 2; ++i) {
        safeGPUAlloc((void **)&(myxa[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(myya[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(myza[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(myvxa[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(myvya[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(myvza[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(thisx_0a[i]), sizeof(int)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(thisy_0a[i]), sizeof(int)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(thisz_0a[i]), sizeof(int)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(time_passeda[i]), sizeof(int)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(updatea[i]), sizeof(int)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(uraya[i]), sizeof(double)*nbeams/2*nrays, i);
        safeGPUAlloc((void **)&(uray_inita[i]), sizeof(double)*nbeams/2*nrays, i);

        moveToAndFromGPU(myxa[i], &myx[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(myya[i], &myy[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(myza[i], &myz[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(myvxa[i], &myvx[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(myvya[i], &myvy[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(myvza[i], &myvz[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(thisx_0a[i], &thisx_0[i*nbeams*nrays/2], sizeof(int)*nbeams/2*nrays, i);
        moveToAndFromGPU(thisy_0a[i], &thisy_0[i*nbeams*nrays/2], sizeof(int)*nbeams/2*nrays, i);
        moveToAndFromGPU(thisz_0a[i], &thisz_0[i*nbeams*nrays/2], sizeof(int)*nbeams/2*nrays, i);
        moveToAndFromGPU(time_passeda[i], &time_passed[i*nbeams*nrays/2], sizeof(int)*nbeams/2*nrays, i);
        moveToAndFromGPU(updatea[i], &update[i*nbeams*nrays/2], sizeof(int)*nbeams/2*nrays, i);
        moveToAndFromGPU(uraya[i], &uray_arr[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
        moveToAndFromGPU(uray_inita[i], &uray_init[i*nbeams*nrays/2], sizeof(double)*nbeams/2*nrays, i);
    }
    */
    pthread_t *threads = new pthread_t[2];
    struct threadData **d = new struct threadData *[2];
    int cpu_counter = 0;
    moveToAndFromGPU(&cpu_counter, counter, sizeof(int), 0);
    //printf("%d\n", cpu_counter);

    d[0] = (struct threadData *)malloc(sizeof(struct threadData));
    d[1] = (struct threadData *)malloc(sizeof(struct threadData));
    d[0]->dedendx = dedendx;
    d[0]->dedendy = dedendy;
    d[0]->dedendz = dedendz;
    d[0]->eden = eden;
    d[0]->edep = edep;
    d[1]->dedendx = dedendx;
    d[1]->dedendy = dedendy;
    d[1]->dedendz = dedendz;
    d[1]->eden = eden;
    d[1]->edep = edep;
    d[1]->etemp = etemp;

    d[0]->etemp = etemp;
    //double *ttt = NULL;
    //safeGPUAlloc((void **)&ttt, sizeof(double)*nbeams*nrays*nt*3, 0);
    for (int i = 0; i < 2; ++i) {
        d[i]->myx = myx;
        d[i]->myy = myy;
        d[i]->myz = myz;
        d[i]->myvx = myvx;
        d[i]->myvy = myvy;
        d[i]->myvz = myvz;
        d[i]->thisx = thisx_0;
        d[i]->thisy = thisy_0;
        d[i]->thisz = thisz_0;
        d[i]->uray = uray_arr;
        d[i]->uray0 = uray_init;
        //d[i]->x = ttt;
        /*
        d->myx = myxa[currThread];
        d->myy = myya[currThread];
        d->myz = myza[currThread];
        d->myvx = myvxa[currThread];
        d->myvy = myvya[currThread];
        d->myvz = myvza[currThread];
        d->thisx = thisx_0a[currThread];
        d->thisy = thisy_0a[currThread];
        d->thisz = thisz_0a[currThread];
        d->uray = uraya[currThread];
        d->uray0 = uray_inita[currThread];
        */
        d[i]->time_passed = time_passed;
        d[i]->update = update;
        d[i]->counter = &cpu_counter;
        d[i]->gIndex = i;
    }
    pthread_create(&threads[0], NULL, runKer, d[0]); 
    pthread_create(&threads[1], NULL, runKer, d[1]);
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    int edepcs = chunk_size+2;
    int edepcs2 = edepcs*edepcs;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int n = 0; n < 8; ++n) {
        int it = n&1;
        int jt = (n&2)>>1;
        int kt = (n&4)>>2;
        for (int i = nx*it; i < nx+2*it; ++i) {
            int il = i%chunk_size*(1-it) + (chunk_size+i-nx)*it;
            for (int j = ny*jt; j < ny+2*jt; ++j) {
                int jl = j%chunk_size*(1-jt) + (chunk_size+j-ny)*jt;
                for (int k = nz*kt; k < nz+2*kt; ++k) {
                    int kl = k%chunk_size*(1-kt) + (chunk_size+k-nz)*kt;
                    int index1 = ((i*(1-it) + (nx-1)*it)/(chunk_size))*nchunksy*nchunksz+(j*(1-jt) + (ny-1)*jt)/(chunk_size)*nchunksz+(k*(1-kt) + (nz-1)*kt)/chunk_size;
                    int index2 = il*edepcs2+jl*edepcs+kl;
                    double temp = 0;
                    if ((kl == 0 && k != 0) || (kl == 1 && k != 1) && !kt) {
                        temp += edep[index1-1][(il)*edepcs2+(jl)*edepcs+(chunk_size+kl)];
                    }
                    if ((jl == 0 && j != 0) || (jl == 1 && j != 1) && !jt) {
                        temp += edep[index1-nchunksz][(il)*edepcs2+(chunk_size+jl)*edepcs+(kl)];
                    }
                    if (((kl == 0 && k != 0) || (kl == 1 && k != 1)) && ((jl == 0 && j != 0) || (jl == 1 && j != 1)) && !kt && !jt) {
                        temp += edep[index1-nchunksz-1][(il)*edepcs2+(chunk_size+jl)*edepcs+(chunk_size+kl)];
                    }
                    if ((il == 0 && i != 0) || (il == 1 && i != 1) && !it) {
                        temp += edep[index1-nchunksz*nchunksy][(chunk_size+il)*edepcs2+(jl)*edepcs+(kl)];
                    }
                    if (((il == 0 && i != 0) || (il == 1 && i != 1)) && ((kl == 0 && k != 0) || (kl == 1 && k != 1)) && !it && !kt) {
                        temp += edep[index1-nchunksz*nchunksy-1][(chunk_size+il)*edepcs2+(jl)*edepcs+(chunk_size+kl)];
                    }
                    if (((il == 0 && i != 0) || (il == 1 && i != 1)) && ((jl == 0 && j != 0) || (jl == 1 && j != 1)) && !it && !jt) {
                        temp += edep[index1-nchunksz*nchunksy-nchunksz][(chunk_size+il)*edepcs2+(chunk_size+jl)*edepcs+(kl)];
                    }
                    if (((il == 0 && i != 0) || (il == 1 && i != 1)) && ((jl == 0 && j != 0) || (jl == 1 && j != 1))
                        && ((kl == 0 && k != 0) || (kl == 1 && k != 1)) && !it && !kt && !jt) {
                        temp += edep[index1-nchunksz*nchunksy-nchunksz-1][(chunk_size+il)*edepcs2+(chunk_size+jl)*edepcs+(chunk_size+kl)];
                    }
                    edep[index1][index2] += temp;
                    edep2[i][j][k] = edep[index1][index2];
                }
            }
        }
    }
    //printf("%lf\n", edep2[101][70][1]);
}

int main(int argc, char **argv) {
    struct timeval time1, time2, time3, total;

#ifdef USE_OPENMP
    if (argc <= 1) {
        cout << "Usage ./cbet-gpu <number of threads>" << endl;
    }
    omp_set_num_threads(atoi(argv[1]));
#endif
    pthread_barrier_init(&b, NULL, 2);
    //////////////////cout << nthreads << endl;
    // Load data from files
    vector<double> r_data(nr), te_data(nr), ne_data(nr);


    fstream file;
    file.open("./s83177_wCBET_t301_1p5ns_te.txt");
    for (unsigned i = 0; i < nr; ++i)
        file >> r_data[i] >> te_data[i];
    file.close();
    file.open("./s83177_wCBET_t301_1p5ns_ne.txt");
    for (unsigned i = 0; i < nr; ++i) {
        file >> r_data[i] >> ne_data[i];
    }
    file.close();

    gettimeofday(&time1, NULL);

    Array3D edep2(boost::extents[nx+2][ny+2][nz+2]);

    //double *x = new double[nx];
    //double *y = new double[ny];
    //double *z = new double[nz];
    double **eden = NULL;
    double **wpe = NULL;
    double **edep = NULL;
/*
    double **eden = (double **)malloc(sizeof(double *)*nchunks);
    // double **etemp = (double **)malloc(sizeof(double *)*nchunks);
    double **edep = (double **)malloc(sizeof(double *)*nchunks);
    double **wpe = (double **)malloc(sizeof(double *)*nchunks);
    for (int i = 0; i < nchunks; ++i) {
        eden[i] = new double[c3];
        // etemp[i] = new double[c3];
        wpe[i] = new double[c3];
        edep[i] = new double[(int)pow((chunk_size+2), 3)];
    }
*/
    /*vector<double> spanx = span(zmin, zmax, nz);
    for (int i = 0; i < nx; ++i) {
        x[i] = spanx[i];
        y[i] = spanx[i];
        z[i] = spanx[i];
    }*/


    /* Calculate the electron density using a function of x and z, as desired. */
    /* sz: Initialize eden and machnum using xz, no dependence among iterations */
/*
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                int idx1 = (i/chunk_size)*nchunksy*nchunksz+(j/chunk_size)*nchunksz+(k/chunk_size);
                int idx2 = (i%chunk_size)*chunk_size*chunk_size+(j%chunk_size)*chunk_size+k%chunk_size;
                double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                eden[idx1][idx2] = interp(ne_data, r_data, tmp);
                // etemp[idx1][idx2] = 5.2e-5 * 10.0 / pow(interp(te_data, r_data, tmp), 1.5);
                // etemp[idx1][idx2] = interp(te_data, r_data, tmp);
                wpe[idx1][idx2] = sqrt((pow(omega, 2) - pow(sqrt(eden[idx1][idx2]*1e6*pow(ec,2)/((double)me*e0)), 2)) /
                        pow(c, 2.0));
                //wpe[idx1][idx2] = sqrt(eden[idx1][idx2]*1e6*pow(ec,2)/((double)me*e0));
                //if (i == 69 && j == 67 && k == 20) {
                  // printf("%lf\n", wpe[idx1][idx2]);
                //}
            }
        }
    }
*/


    // sz: Does C++ always initialize to 0?
    //std::fill_n(edep.data(), edep.num_elements(), 0);

    gettimeofday(&time2, NULL);

    rayTracing(te_data, r_data, ne_data, eden, edep, wpe, NULL, edep2);
    /*for (unsigned i = 0; i < 1; ++i) {
        for (unsigned j = 0; j < 1; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                int idx1 = (i/(chunk_size))*nchunksy*nchunksz+(j/(chunk_size))*nchunksz+(k/(chunk_size));
                int idx2 = (i%(chunk_size))*(chunk_size)*(chunk_size)+(j%(chunk_size))*(chunk_size)+k%(chunk_size);
                edep2[i][j][k] = edep[idx1][idx2];
                //printf("%d %d %d\n", k, idx1, idx2);
            }
         }
    }*/
    Array3D edepavg(boost::extents[nx][nz][nz]);
#if 1
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                edepavg[i][j][k] =
                    (edep2[i][j][k] + edep2[i+1][j][k] + edep2[i+2][j][k] +
                     edep2[i][j+1][k] + edep2[i+1][j+1][k] + edep2[i+2][j+1][k] +
                     edep2[i][j+2][k] + edep2[i+1][j+2][k] + edep2[i+2][j+2][k] +
                     edep2[i][j][k+1] + edep2[i+1][j][k+1] + edep2[i+2][j][k+1] +
                     edep2[i][j+1][k+1] + edep2[i+1][j+1][k+1] + edep2[i+2][j+1][k+1] +
                     edep2[i][j+2][k+1] + edep2[i+1][j+2][k+1] + edep2[i+2][j+2][k+1] +
                     edep2[i][j][k+2] + edep2[i+1][j][k+2] + edep2[i+2][j][k+2] +
                     edep2[i][j+1][k+2] + edep2[i+1][j+1][k+2] + edep2[i+2][j+1][k+2] +
                     edep2[i][j+2][k+2] + edep2[i+1][j+2][k+2] + edep2[i+2][j+2][k+2]) / 27;
            }
        }
    }
#endif
    gettimeofday(&time3, NULL);
    //save2Hdf5(x, y, z, edepavg);
#ifdef PRINT
    print(std::cout, edep2);
    // print2d(std::cout, edep, nchunks, (int)pow((chunk_size+2), 3)); 
#endif

    timersub(&time3, &time1, &total);
    timersub(&time3, &time2, &time3);
    timersub(&time2, &time1, &time2);
#if 1
#ifndef PRINT
    printf("Init %ld.%06ld\nRay tracing %ld.%06ld\n"
           "Total %ld.%06ld\n",
           time2.tv_sec, time2.tv_usec,
           time3.tv_sec, time3.tv_usec,
           total.tv_sec, total.tv_usec);
#endif
#endif

    return 0;
}
