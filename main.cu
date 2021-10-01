
#include "def.cuh"

using namespace H5;

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

void print(std::ostream& os, const double& x)
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

vector<double> span(double minimum, double maximum, unsigned len) {
    double step = (maximum - minimum) / (len - 1), curr = minimum;
    vector<double> ret(len);
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
    catch(H5::Exception& error )
        {
            error.printErrorStack();
            return -1;
        }
    return 0;  // successfully terminated
}

void rayTracing(Array3D eden, Array3D etemp, 
        double *ne_profile, double *edep, int *marked, int *boxes, double *coverage, int *counter) {

    struct timeval time1, time2, time3, time4, total;
    gettimeofday(&time1, NULL);

    vector<double> phase_r = span(0.0, 0.1, 2001);
    vector<double> pow_r(2001);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < 2001; ++i) {
        pow_r[i] = exp(-1*pow(pow((phase_r[i]/sigma),2), (5.0/2.0)));
    }
    //printf("Starting CUDA code\n");

    double **dev_beam_norm = new double *[nGPUs];
    double **dev_bbeam_norm = new double *[nGPUs];
    double **dev_ne_profile = new double *[nGPUs];
    double **dev_te_profile = new double *[nGPUs];
    double **dev_r_profile = new double *[nGPUs];
    double **dev_pow_r = new double *[nGPUs];
    double **dev_phase_r = new double *[nGPUs];

	double **dev_eden = new double *[nGPUs];
 	double **dev_etemp = new double *[nGPUs];

    int **dev_marked = new int *[nGPUs];
    int **dev_boxes = new int *[nGPUs];

  	double **dev_coverage = new double *[nGPUs];

    double *better_beam_norm = new double[nbeams*4];
    for (int b = 0; b < nbeams; ++b) {
        double theta1 = acos(beam_norm[b][2]);
        double theta2 = atan2(beam_norm[b][1]*focal_length, beam_norm[b][0]*focal_length);
        better_beam_norm[4*b] = cos(theta1);
        better_beam_norm[4*b+1] = sin(theta1);
        better_beam_norm[4*b+2] = cos(theta2);
        better_beam_norm[4*b+3] = sin(theta2);
    }

    double **dev_edep = new double *[nGPUs];
    double **edep_per_GPU = new double *[nGPUs];
    for (int i = 0; i < nGPUs; ++i) {
        edep_per_GPU[i] = (double *)malloc(sizeof(double)*edep_size);

        safeGPUAlloc((void **)&dev_beam_norm[i], sizeof(double)*3*nbeams, i);
        safeGPUAlloc((void **)&dev_bbeam_norm[i], sizeof(double)*4*nbeams, i);
        safeGPUAlloc((void **)&dev_pow_r[i], sizeof(double)*2001, i);
        safeGPUAlloc((void **)&dev_phase_r[i], sizeof(double)*2001, i);
        safeGPUAlloc((void **)&dev_ne_profile[i], sizeof(double)*nr, i);
        safeGPUAlloc((void **)&dev_te_profile[i], sizeof(double)*nr, i);
        safeGPUAlloc((void **)&dev_r_profile[i], sizeof(double)*nr, i);
        safeGPUAlloc((void **)&dev_edep[i], sizeof(double)*edep_size, i);
        //safeGPUAlloc((void **)&dev_marked[i], sizeof(int)*nx*ny*nz*2020, i);
        safeGPUAlloc((void **)&dev_boxes[i], sizeof(int)*nbeams*nrays*ncrossings*3, i);
        safeGPUAlloc((void **)&dev_coverage[i], sizeof(double)*nbeams*nrays*ncrossings, i);
		    safeGPUAlloc((void **)&dev_eden[i], sizeof(double)*nx*ny*nz, i);
		    safeGPUAlloc((void **)&dev_etemp[i], sizeof(double)*nx*ny*nz, i);
    
        moveToAndFromGPU(dev_beam_norm[i], &(beam_norm[0][0]), sizeof(double)*3*nbeams, i);
        moveToAndFromGPU(dev_bbeam_norm[i], &(better_beam_norm[0]), sizeof(double)*4*nbeams, i);
        moveToAndFromGPU(dev_pow_r[i], &(pow_r[0]), sizeof(double)*2001, i);
        moveToAndFromGPU(dev_phase_r[i], &(phase_r[0]), sizeof(double)*2001, i);
        /*moveToAndFromGPU(dev_ne_profile[i], &(ne_profile[0]), sizeof(double)*nr, i);
        moveToAndFromGPU(dev_te_profile[i], &(te_profile[0]), sizeof(double)*nr, i);
        moveToAndFromGPU(dev_r_profile[i], &(r_profile[0]), sizeof(double)*nr, i);*/
		    moveToAndFromGPU(dev_eden[i], &(eden[0][0][0]), sizeof(double)*nx*ny*nz, i);
		    moveToAndFromGPU(dev_etemp[i], &(etemp[0][0][0]), sizeof(double)*nx*ny*nz, i);
    }

    gettimeofday(&time2, NULL);

    double grad_const = pow(c, 2) / (2.0 * ncrit) * dt * 0.5;
    double dedx_const = grad_const / xres;
    double dedy_const = grad_const / yres;
    double dedz_const = grad_const / zres;

    dim3 nblocks(nbeams/nGPUs, threads_per_beam/threads_per_block, 1);
    //printf("%d %d\n", nbeams/nGPUs, threads_per_beam);
    //printf("%d\n", nindices);

    // We put the launches in their own loops for timing purposes
#ifdef USE_OPENMP
#pragma omp parallel for num_threads (nGPUs)
#endif
    for (int i = 0; i < nGPUs; ++i) { 
        cudaSetDevice(1);
        launch_ray_XYZ<<<nblocks, threads_per_block>>>(i, nindices, dev_eden[i], dev_etemp[i],
            dev_edep[i], dev_bbeam_norm[i],
            dev_beam_norm[i], dev_pow_r[i], dev_phase_r[i],
            dedx_const, dedy_const, dedz_const, marked, dev_boxes[i], dev_coverage[i], counter);
        cout << cudaDeviceSynchronize() << endl;
    }
    
    for (int i = 0; i < nGPUs; ++i) {
        moveToAndFromGPU(edep_per_GPU[i], dev_edep[i], sizeof(double)*edep_size, i);
        moveToAndFromGPU(boxes, dev_boxes[i], sizeof(int)*nbeams*nrays*ncrossings*3, i);
        moveToAndFromGPU(coverage, dev_coverage[i], sizeof(double)*nbeams*nrays*ncrossings, i);
        cudaFree(dev_pow_r[i]);
        cudaFree(dev_phase_r[i]);
        cudaFree(dev_beam_norm[i]);
        cudaFree(dev_bbeam_norm[i]);
        cudaFree(dev_te_profile[i]);
        cudaFree(dev_ne_profile[i]);
        cudaFree(dev_r_profile[i]);
        cudaFree(dev_edep[i]);
        cudaFree(dev_eden[i]);
        cudaFree(dev_etemp[i]);
        cudaFree(dev_coverage[i]);
        cudaFree(dev_boxes[i]);
    }
    delete[] dev_pow_r;
    delete[] dev_phase_r;
    delete[] dev_beam_norm;
    delete[] dev_bbeam_norm;
    delete[] dev_te_profile;
    delete[] dev_ne_profile;
    delete[] dev_r_profile;
    delete[] dev_edep;
    delete[] dev_eden;
    delete[] dev_etemp;
    delete[] dev_coverage;
    delete[] dev_boxes;

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

    cudaDeviceReset();

    gettimeofday(&time4, NULL);
    timersub(&time4, &time1, &total);
    timersub(&time4, &time3, &time4);
    timersub(&time3, &time2, &time3);
    timersub(&time2, &time1, &time2);
#ifndef PRINT
    printf("rt: Init %ld.%06ld\nTracing %ld.%06ld\nCombining %ld.%06ld\n"
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

    // r_data is the radius profile, ne_data is the electron profile
    // and te_data is the temperature profile
    vector<double> r_data(nr), te_data(nr), ne_data(nr);

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

    // 1200 here is temporary, need to come up with bound
    cudaMallocManaged(&marked, sizeof(int)*nx*ny*nz*1200);
    cudaMallocManaged(&counter, sizeof(int)*nx*ny*nz);

    Array4I boxes(boost::extents[nbeams][nrays][ncrossings][3]);
    Array3D area_coverage(boost::extents[nbeams][nrays][ncrossings]);
    
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {
                double tmp = sqrt(pow(i*dx+xmin, 2) + pow(j*dy+ymin, 2) + pow(k*dz+zmin, 2));
                eden[i][j][k] = interp(ne_data, r_data, tmp);
                //etemp[i][j][k] = 5.2e-5 * 10.0 / pow(interp(te_data, r_data, tmp), 1.5);
                etemp[i][j][k] = interp(te_data, r_data, tmp);
                //wpe[i][j][k] = sqrt((pow(omega, 2) - pow(sqrt(eden[i][j][k]*1e6*pow(ec,2)/((double)me*e0)), 2)) /
                 //       pow(c, 2.0));
                //wpe[i][j][k] = sqrt(eden[i][j][k]*1e6*pow(ec,2)/((double)me*e0));
            }
        }
    }
    rayTracing(eden, etemp, NULL, &edep[0][0][0], marked, &boxes[0][0][0][0], &area_coverage[0][0][0], counter);
    // Below is the code for writing to the hdf5 file
    /*
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
*/
    // For quick testing
#ifdef PRINT
    print(std::cout, edep);
#endif
    return 0;
}
