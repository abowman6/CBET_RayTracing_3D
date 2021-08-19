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
            //cout << "Storing data on GPU: " << i << endl;
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

void print1d(ostream& f, vector<double> array) {
    std::cout.precision(8);
    for (vector<double>::size_type i = 0; i < array.size(); i++ )
        f << array[i] << ' ';
    f << std::endl;
}

void print2d(ostream& f, vector<vector<double>> array) {
    std::cout.precision(8);
    for (vector<vector<double>>::size_type i = 0; i < array.size(); i++ ) {
        for (vector<double>::size_type j = 0; j < array[i].size(); j++ ) {
            f << array[i][j] << ' ';
        }
        f << std::endl;
    }
}

void print(std::ostream& os, const double& x)
{
  os << x;
}

template <typename Array>
void print(std::ostream& os, const Array& A)
{
  typename Array::const_iterator i;
  os << "[";
  for (i = A.begin(); i != A.end(); ++i) {
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

void rayTracing(Array3D& x, Array3D& y, Array3D& z,
                Array3D& eden, Array3D& edep, Array3D& wpe, Array3D& etemp)
{
    Array3D dedendx(boost::extents[nx][nz][nz]),
        dedendy(boost::extents[nx][nz][nz]),
        dedendz(boost::extents[nx][nz][nz]);

    /* Calculate the gradients of electron density w.r.t. x and z */
    // Central differences
    /* sz: Initialize dedendx/z using eden and xz, no dependence among iterations
       The last row/col needs to be treated differently
     */
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (unsigned i = 1; i < nx - 1; ++i){
        for (unsigned j = 1; j < ny - 1; ++j){
            for (unsigned k = 1; k < nz - 1; ++k){
                // Can we optimize like this?
                dedendx[i][j][k] = 0.5 * (eden[i+1][j][k] - eden[i-1][j][k]) / xres;
                dedendy[i][j][k] = 0.5 * (eden[i][j+1][k] - eden[i][j-1][k]) / yres;
                dedendz[i][j][k] = 0.5 * (eden[i][j][k+1] - eden[i][j][k-1]) / zres;

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
            dedendx[0][i][j] = dedendx[1][i][j];
            dedendy[i][0][j] = dedendy[i][1][j];
            dedendz[i][j][0] = dedendz[i][j][1];
            dedendx[nx-1][i][j] = dedendx[nx-2][i][j];
            dedendy[i][ny-1][j] = dedendy[i][ny-2][j];
            dedendz[i][j][nz-1] = dedendz[i][j][nz-2];
        }
    }
    // sz: depends on eden, which may need to reorder
    // ArrayXXd wpe = (eden*1e6*pow(ec,2)/((double)me*e0)).sqrt();

    vector<double> phase_r = span(0.0, 0.1, 2001);
    vector<double> pow_r(2001);
    for (unsigned i = 0; i < 2001; ++i) {
        pow_r[i] = exp(-1*pow(pow((phase_r[i]/sigma),2), (5.0/2.0)));
    }

    double beam_centers[60][3];
    for (unsigned i = 0; i < 60; i++)
        for (unsigned j = 0; j < 3; j++)
            beam_centers[i][j] = beam_norm[i][j] * focal_length;

    cudaError_t e = cudaGetDeviceCount(&numGPUs);
    if (e != 0) {
        cout << cudaGetErrorString(e) << endl;
        return;
    }
    usable = new bool[numGPUs];
    int numUsableGPUs = numGPUs;

    // for simplicity allow all devices to access each other's memory
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

    double *devx,*devy,*devz,*dwpe,*devdedendx,*devdedendy,*devdedendz,*dedep,
        *deden,*detemp,*dbeam_centers,*dbeam_norm,*dpow_r,*dphase_r;

    int xidx = initializeOnGPU((void **)&devx, sizeof(double)*nz*ny*nx);
    int yidx = initializeOnGPU((void **)&devy, sizeof(double)*nz*ny*nx);
    int zidx = initializeOnGPU((void **)&devz, sizeof(double)*nz*ny*nx);
    int dedxidx = initializeOnGPU((void **)&devdedendx, sizeof(double)*nz*ny*nx);
    int dedyidx = initializeOnGPU((void **)&devdedendy, sizeof(double)*nz*ny*nx);
    int dedzidx = initializeOnGPU((void **)&devdedendz, sizeof(double)*nz*ny*nx);
    int wpeidx = initializeOnGPU((void **)&dwpe, sizeof(double)*nz*ny*nx);
    int edenidx = initializeOnGPU((void **)&deden, sizeof(double)*nz*ny*nx);
    int etempidx = initializeOnGPU((void **)&detemp, sizeof(double)*nz*ny*nx);
    int bcidx = initializeOnGPU((void **)&dbeam_centers, sizeof(double)*180);
    int bnidx = initializeOnGPU((void **)&dbeam_norm, sizeof(double)*180);
    int poridx = initializeOnGPU((void **)&dpow_r, sizeof(double)*2001);
    int phridx = initializeOnGPU((void **)&dphase_r, sizeof(double)*2001);
    int edepidx = initializeOnGPU((void **)&dedep, sizeof(double)*(nx+2)*(ny+2)*(nz+2));

    moveToAndFromGPU(devx, &(x[0][0][0]), sizeof(double)*nz*ny*nx, xidx);
    moveToAndFromGPU(devy, &(y[0][0][0]), sizeof(double)*nz*ny*nx, yidx);
    moveToAndFromGPU(devz, &(z[0][0][0]), sizeof(double)*nz*ny*nx, zidx);
    moveToAndFromGPU(devdedendx, &(dedendx[0][0][0]), sizeof(double)*nz*ny*nx, dedxidx);
    moveToAndFromGPU(devdedendy, &(dedendy[0][0][0]), sizeof(double)*nz*ny*nx, dedyidx);
    moveToAndFromGPU(devdedendz, &(dedendz[0][0][0]), sizeof(double)*nz*ny*nx, dedzidx);
    moveToAndFromGPU(dwpe, &(wpe[0][0][0]), sizeof(double)*nz*ny*nx, wpeidx);
    moveToAndFromGPU(deden, &(eden[0][0][0]), sizeof(double)*nz*ny*nx, edenidx);
    moveToAndFromGPU(detemp, &(etemp[0][0][0]), sizeof(double)*nz*ny*nx, etempidx);
    moveToAndFromGPU(dbeam_centers, &(beam_centers[0][0]), sizeof(double)*180, bcidx);
    moveToAndFromGPU(dbeam_norm, &(beam_norm[0][0]), sizeof(double)*180, bnidx);
    moveToAndFromGPU(dpow_r, &(pow_r[0]), sizeof(double)*2001, poridx);
    moveToAndFromGPU(dphase_r, &(phase_r[0]), sizeof(double)*2001, phridx);
    
    cout << nindices << endl;    
    cout << threads_per_beam << endl;    

    dim3 nblocks(nbeams, threads_per_beam/threads_per_block, 1);
    int rays_per_thread = ceil(nrays/(float)((nthreads/nbeams)/threads_per_block*threads_per_block));
    // ab: the first argument will be needed for splitting the work across 
    // GPUs so I leave it in.
    launch_ray_XYZ<<<nblocks, threads_per_block>>>(1, nindices, devx, devy, devz, dwpe, devdedendx, devdedendy,
        devdedendz, dedep, deden, detemp, dbeam_centers, dbeam_norm, dpow_r, dphase_r);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();
    cudaMemcpy(&(edep[0][0][0]), dedep, sizeof(double)*(nx+2)*(nz+2)*(ny+2), cudaMemcpyDefault);
}
int main(int argc, char **argv) {
    struct timeval time1, time2, time3, total;

#ifdef USE_OPENMP
    if (argc <= 1) {
        cout << "Usage ./cbet-gpu <number of threads>" << endl;
    }
    omp_set_num_threads(atoi(argv[1]));
#endif

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

    /* Define 2D arrays that will store data for electron density,
       derivatives of e_den, and x/z */
    Array3D eden(boost::extents[nx][nz][nz]), x(boost::extents[nx][nz][nz]),
        y(boost::extents[nx][nz][nz]), z(boost::extents[nx][nz][nz]),
        etemp(boost::extents[nx][nz][nz]),
        edep(boost::extents[nx+2][nz+2][nz+2]),
        wpe(boost::extents[nx][nz][nz]);

    vector<double> spanx = span(zmin, zmax, nz);

    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                x[i][j][k] = spanx[i];
                y[i][j][k] = spanx[j];
                z[i][j][k] = spanx[k];
            }
        }
    }

    /* Calculate the electron density using a function of x and z, as desired. */
    /* sz: Initialize eden and machnum using xz, no dependence among iterations */
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                double tmp = sqrt(pow(x[i][j][k], 2) + pow(y[i][j][k], 2) + pow(z[i][j][k], 2));
                eden[i][j][k] = interp(ne_data, r_data, tmp);
                etemp[i][j][k] = 5.2e-5 * 10.0 / pow(interp(te_data, r_data, tmp), 1.5);
                // etemp[i][j][k] = interp(te_data, r_data, tmp);
                wpe[i][j][k] = sqrt((pow(omega, 2) - pow(sqrt(eden[i][j][k]*1e6*pow(ec,2)/((double)me*e0)), 2)) /
                        pow(c, 2.0));
                //wpe[i][j][k] = sqrt(eden[i][j][k]*1e6*pow(ec,2)/((double)me*e0));
            }
        }
    }

    // sz: Does C++ always initialize to 0?
    std::fill_n(edep.data(), edep.num_elements(), 0);

    gettimeofday(&time2, NULL);

    rayTracing(x, y, z, eden, edep, wpe, etemp);
    Array3D edepavg(boost::extents[nx][nz][nz]);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                edepavg[i][j][k] =
                    (edep[i][j][k] + edep[i+1][j][k] + edep[i+2][j][k] +
                     edep[i][j+1][k] + edep[i+1][j+1][k] + edep[i+2][j+1][k] +
                     edep[i][j+2][k] + edep[i+1][j+2][k] + edep[i+2][j+2][k] +
                     edep[i][j][k+1] + edep[i+1][j][k+1] + edep[i+2][j][k+1] +
                     edep[i][j+1][k+1] + edep[i+1][j+1][k+1] + edep[i+2][j+1][k+1] +
                     edep[i][j+2][k+1] + edep[i+1][j+2][k+1] + edep[i+2][j+2][k+1] +
                     edep[i][j][k+2] + edep[i+1][j][k+2] + edep[i+2][j][k+2] +
                     edep[i][j+1][k+2] + edep[i+1][j+1][k+2] + edep[i+2][j+1][k+2] +
                     edep[i][j+2][k+2] + edep[i+1][j+2][k+2] + edep[i+2][j+2][k+2]) / 27;
            }
        }
    }
    gettimeofday(&time3, NULL);

    //save2Hdf5(x, y, z, edepavg);
#ifdef PRINT
    print(std::cout, edep);
#endif

    timersub(&time3, &time1, &total);
    timersub(&time3, &time2, &time3);
    timersub(&time2, &time1, &time2);

#ifndef PRINT
    printf("Init %ld.%06ld\nRay tracing %ld.%06ld\n"
           "Total %ld.%06ld\n",
           time2.tv_sec, time2.tv_usec,
           time3.tv_sec, time3.tv_usec,
           total.tv_sec, total.tv_usec);
#endif

    return 0;
}
