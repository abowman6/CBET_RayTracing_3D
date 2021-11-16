
#include "constants.cuh"

using namespace H5;

dtype interp(const vector<dtype> y, const vector<dtype> x, const dtype xp)
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
        while (low < high - 1) { if (x[mid] >= xp) high = mid;
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

void print(std::ostream& os, const dtype& x)
{
  os << x;
}

template <typename Array>
void print(std::ostream& os, const Array& A)
{
  typename Array::const_iterator i;
  os << "[";
  for (i = A.begin(); i != A.end(); i++) {
    print(os, *i);
    if (boost::next(i) != A.end())
      os << ',';
  }
  os << "]" << endl;
}

vector<dtype> span(dtype minimum, dtype maximum, unsigned len) {
    dtype step = (maximum - minimum) / (len - 1), curr = minimum;
    vector<dtype> ret(len);
    for (unsigned i = 0; i < len; ++i) {
        ret[i] = curr;
        curr += step;
    }
    return ret;
}

/*
   Save multi_array to hdf5 file
 */
int save2Hdf5(Array3D& x, Array3D& y, Array3D& z, Array3D& edepavg)
{
    const H5std_string  FILE_NAME("edep.hdf5");
#ifndef DOUBLE
    auto type = H5::PredType::NATIVE_FLOAT;
#else
    auto type = H5::PredType::NATIVE_DOUBLE;
#endif

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
            H5::IntType datatype(type);
            datatype.setOrder(H5T_ORDER_LE);
            /*
             * Create a new dataset within the file using defined dataspace and
             * datatype and default dataset creation properties.
             */
            H5::DataSet dataset = file.createDataSet("/Coordinate_x", datatype, dataspace);
            dataset.write(x.data(), type);
            dataset = file.createDataSet("/Coordinate_y", datatype, dataspace);
            dataset.write(y.data(), type);
            dataset = file.createDataSet("/Coordinate_z", datatype, dataspace);
            dataset.write(z.data(), type);

            dataset = file.createDataSet("/Edepavg", datatype, dataspace);
            /*
             * Write the data to the dataset using default memory space, file
             * space, and transfer properties.
             */
            dataset.write(edepavg.data(), type);
        }  // end of try block
    // catch failure caused by the H5File operations
    catch(H5::Exception& error )
        {
            error.printErrorStack();
            return -1;
        }
    return 0;  // successfully terminated
}

void rayTracing(Array3D eden, Array3D etemp, 
        dtype *ne_profile, dtype *edep, int *marked, int *boxes, dtype *coverage, int *counter) {

    struct timeval time1, time2, time3, time4, total;
    gettimeofday(&time1, NULL);

    vector<dtype> phase_r = span(0.0, 0.1, 2001);
    vector<dtype> pow_r(2001);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < 2001; ++i) {
        pow_r[i] = exp(-1*pow(pow((phase_r[i]/sigma),2), (5.0/2.0)));
    }

    dtype **dev_beam_norm = new dtype *[nGPUs];
    dtype **dev_pow_r = new dtype *[nGPUs];
    dtype **dev_phase_r = new dtype *[nGPUs];

	  dtype **dev_eden = new dtype *[nGPUs];
 	  dtype **dev_etemp = new dtype *[nGPUs];

    int **dev_marked = new int *[nGPUs];
    //int **dev_boxes = new int *[nGPUs];

  	//dtype **dev_coverage = new dtype *[nGPUs];

    dtype **dev_edep = new dtype *[nGPUs];
    dtype **edep_per_GPU = new dtype *[nGPUs];
    for (int i = 0; i < nGPUs; ++i) {
        edep_per_GPU[i] = (dtype *)malloc(sizeof(dtype)*edep_size);

        safeGPUAlloc((void **)&dev_beam_norm[i], sizeof(dtype)*3*nbeams, i);
        safeGPUAlloc((void **)&dev_pow_r[i], sizeof(dtype)*2001, i);
        safeGPUAlloc((void **)&dev_phase_r[i], sizeof(dtype)*2001, i);
        safeGPUAlloc((void **)&dev_edep[i], sizeof(dtype)*edep_size, i);
        //safeGPUAlloc((void **)&dev_boxes[i], sizeof(int)*nbeams*nrays*ncrossings*3, i);
        //safeGPUAlloc((void **)&dev_coverage[i], sizeof(dtype)*nbeams*nrays*ncrossings, i);
		    safeGPUAlloc((void **)&dev_eden[i], sizeof(dtype)*nx*ny*nz, i);
		    safeGPUAlloc((void **)&dev_etemp[i], sizeof(dtype)*nx*ny*nz, i);
    
        moveToAndFromGPU(dev_beam_norm[i], &(beam_norm[0][0]), sizeof(dtype)*3*nbeams, i);
        moveToAndFromGPU(dev_pow_r[i], &(pow_r[0]), sizeof(dtype)*2001, i);
        moveToAndFromGPU(dev_phase_r[i], &(phase_r[0]), sizeof(dtype)*2001, i);
		    moveToAndFromGPU(dev_eden[i], &(eden[0][0][0]), sizeof(dtype)*nx*ny*nz, i);
		    moveToAndFromGPU(dev_etemp[i], &(etemp[0][0][0]), sizeof(dtype)*nx*ny*nz, i);
    }

    gettimeofday(&time2, NULL);

    dim3 nblocks(nbeams/nGPUs, threads_per_beam/threads_per_block, 1);

#ifdef USE_OPENMP
#pragma omp parallel for num_threads (nGPUs)
#endif
    for (int i = 0; i < nGPUs; ++i) { 
        cudaSetDevice(i);
        launch_ray_XYZ<<<nblocks, threads_per_block>>>(i, nindices, dev_eden[i], dev_etemp[i],
            dev_edep[i], NULL,
            dev_beam_norm[i], dev_pow_r[i], dev_phase_r[i],
            marked, boxes, coverage, counter);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != 0) {
          cout << e << endl;
        }
    }
    
    for (int i = 0; i < nGPUs; ++i) {
        moveToAndFromGPU(edep_per_GPU[i], dev_edep[i], sizeof(dtype)*edep_size, i);
        //moveToAndFromGPU(boxes, dev_boxes[i], sizeof(int)*nbeams*nrays*ncrossings*3, i);
        //moveToAndFromGPU(coverage, dev_coverage[i], sizeof(dtype)*nbeams*nrays*ncrossings, i);
        cudaFree(dev_pow_r[i]);
        cudaFree(dev_phase_r[i]);
        cudaFree(dev_beam_norm[i]);
        cudaFree(dev_edep[i]);
        cudaFree(dev_eden[i]);
        cudaFree(dev_etemp[i]);
        //cudaFree(dev_coverage[i]);
        //cudaFree(dev_boxes[i]);
    }
    delete[] dev_pow_r;
    delete[] dev_phase_r;
    delete[] dev_beam_norm;
    delete[] dev_edep;
    delete[] dev_eden;
    delete[] dev_etemp;
    //delete[] dev_coverage;
    //delete[] dev_boxes;

    gettimeofday(&time3, NULL);
    for (int l = 0; l < nGPUs; ++l) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
        for (long i = 0; i < nx+2; ++i) {
            for (long j = 0; j < ny+2; ++j) {
                for (long k = 0; k < nz+2; ++k) {
                    edep[i*(ny+2)*(nz+2) + j*(nz+2) + k] += edep_per_GPU[l][i*(ny+2)*(nz+2) + j*(nz+2) + k];
                }
            }
        }
    }
    for (int i = 0; i < nGPUs; ++i) {
        free(edep_per_GPU[i]);
    }

    delete[] edep_per_GPU;

    gettimeofday(&time4, NULL);
    timersub(&time4, &time1, &total);
    timersub(&time4, &time3, &time4);
    timersub(&time3, &time2, &time3);
    timersub(&time2, &time1, &time2);
#ifndef PRINT
    printf("Moving %ld.%06ld\nTracing %ld.%06ld\nCombining %ld.%06ld\n"
           "Total %ld.%06ld\n",
           time2.tv_sec, time2.tv_usec,
           time3.tv_sec, time3.tv_usec,
           time4.tv_sec, time4.tv_usec,
           total.tv_sec, total.tv_usec);
#endif
}

int main(int argc, char **argv) {

    if (argc <= 1) {
    } else {
        omp_set_num_threads(atoi(argv[1]));
    }
    struct timeval time0, time1, time2, time3;
    gettimeofday(&time0, NULL);
    cudaDeviceEnablePeerAccess(1,0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0,0);
    cudaSetDevice(0);
    gettimeofday(&time1, NULL);
 
    // r_data is the radius profile, ne_data is the electron profile
    // and te_data is the temperature profile
    vector<dtype> r_data(nr), te_data(nr), ne_data(nr);

    // Load data from files
    fstream file;
    file.open("./s83177_wCBET_t301_1p5ns_te.txt");
    for (unsigned i = 0; i < nr; ++i)
        file >> r_data[i] >> te_data[i];
    file.flush();
    file.close();
    file.open("./s83177_wCBET_t301_1p5ns_ne.txt");
    for (unsigned i = 0; i < nr; ++i) {
        file >> r_data[i] >> ne_data[i];
    }
    file.flush();
    file.close();

    Array3D edep(boost::extents[nx+2][ny+2][nz+2]),
		  etemp(boost::extents[nx][nz][nz]),
      eden(boost::extents[nx][nz][nz]);

    int *marked, *counter;
    int *boxes;
    dtype *area_coverage;

    // 1200 here is temporary, need to come up with bound
    cudaMallocManaged(&marked, sizeof(int)*nx*ny*nz*marked_const);
    cudaMallocManaged(&counter, sizeof(int)*nx*ny*nz);
  
    cudaMallocManaged(&boxes, sizeof(int)*nrays*nbeams*ncrossings*3);
    cudaMallocManaged(&area_coverage, sizeof(dtype)*nrays*nbeams*ncrossings);

    //Array4I boxes(boost::extents[nbeams][nrays][ncrossings][3]);
    //Array3D area_coverage(boost::extents[nbeams][nrays][ncrossings]);
    
    gettimeofday(&time2, NULL);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                dtype tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                eden[i][j][k] = interp(ne_data, r_data, tmp);
                etemp[i][j][k] = interp(te_data, r_data, tmp);
            }
        }
    }
    gettimeofday(&time3, NULL);
    timersub(&time3, &time2, &time3);
    timersub(&time2, &time1, &time2);
    timersub(&time1, &time0, &time1);
    printf("CUDA Init %ld.%06ld\n", time1.tv_sec, time1.tv_usec);
    printf("Malloc Managed %ld.%06ld\n", time2.tv_sec, time2.tv_usec);
    printf("Setup %ld.%06ld\n", time3.tv_sec, time3.tv_usec);
    rayTracing(eden, etemp, NULL, &edep[0][0][0], marked, boxes, area_coverage, counter);
    int max = 0;
    for (int i = 0; i < nx*ny*nz; ++i) {
      if (counter[i] > max) {
        max = counter[i];
      }
    }
    printf("%d\n", max);
    // Below is the code for writing to the hdf5 file
#if 0
    struct timeval time3, time4;
    gettimeofday(&time3, NULL);
    Array3D edepavg(boost::extents[nx][ny][nz]);
    Array3D x(boost::extents[nx][ny][nz]);
    Array3D y(boost::extents[nx][ny][nz]);
    Array3D z(boost::extents[nx][ny][nz]);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                x[i][j][k] = i*dx+xmin;
                y[i][j][k] = j*dy+ymin;
                z[i][j][k] = k*dz+zmin;
            }
        }
    }

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

    save2Hdf5(x, y, z, edepavg);
    gettimeofday(&time4, NULL);
    timersub(&time4, &time3, &time4);
    printf("Write to hdf5 %ld.%06ld\n", time4.tv_sec, time4.tv_usec);
#endif
    // For quick testing
#ifdef PRINT
    print(std::cout, edep);
#endif
    return 0;
    cudaDeviceReset();
}
