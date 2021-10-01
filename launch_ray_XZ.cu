#include <iostream>
#include "def.cuh"
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ long edep_index(long x, long y, long z) {
    return x*(ny+2)*(nz+2) + y*(nz+2) + z;
}

__device__ int eden_index(int x, int y, int z) {
	return x*ny*nz + y*nz + z;
}

__device__ double square(double x){
    return x*x;
}

// Piecewise linear interpolation
// Use binary search to find the segment
// Ref: https://software.llnl.gov/yorick-doc/qref/qrfunc09.html
__device__ double interp_cuda(double *y, double *x, const double xp, int n)
{
    unsigned low, high, mid;

    if (x[0] <= x[n-1]) {
        // x is increasing
        if (xp <= x[0])
            return y[0];
        else if (xp >= x[n-1])
            return y[n-1];

        low = 0;
        high = n - 1;
        mid = (low + high) >> 1;
        while (low < high - 1) {
            if (x[mid] >= xp)
                high = mid;
            else
                low = mid;
            mid = (low + high) >> 1;
        }

        //assert((xp >= x[mid]) && (xp <= x[mid + 1]));
        return y[mid] +
            (y[mid + 1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    } else {
        // x is decreasing
        if (xp >= x[0])
            return y[0];
        else if (xp <= x[n-1])
            return y[n-1];

        low = 0;
        high = n - 1;
        mid = (low + high) >> 1;
        while (low < high - 1) {
            if (x[mid] <= xp)
                low = mid;
            else
                high = mid;
            mid = (low + high) >> 1;
        }

        //assert((xp <= x[mid]) && (xp >= x[mid + 1]));
        return y[mid] +
            (y[mid +1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    }
}

__device__ 
bool init(int beam, int pre_raynum, double &x_init, double &y_init, double &z_init,
            double &uray_init, const double *beam_norm, double *pow_r, double *phase_r) {

    int zones_spanned = ceil((beam_max_x-beam_min_x)/xres);
    int b1 = pre_raynum/(rays_per_zone*rays_per_zone);
    int b2 = pre_raynum%(rays_per_zone*rays_per_zone);
    int ry = b1/(zones_spanned)*rays_per_zone + b2/rays_per_zone;
    int rx = b1%(zones_spanned)*rays_per_zone + b2%rays_per_zone;
    int raynum = ry*nrays_x+rx;

    //if (raynum == 51) printf("%d\n", pre_raynum);

    x_init = beam_min_x;
    for (int i = 0; i < (raynum % nrays_x); i++) {
        x_init += (beam_max_x - beam_min_x) / (nrays_x - 1);
    }

    // ab: this should be faster but in order to agree with CPU I do it the way above
    //x_init = (raynum % nrays_x) * (beam_max_x - beam_min_x) / (nrays_x - 1) + beam_min_x;
    x_init += dx/2;

    y_init = beam_min_x;
    for (int i = 0; i < (raynum / nrays_x); i++) {
        y_init += (beam_max_x - beam_min_x) / (nrays_y - 1);
    }

    // ab: same here
    //y_init = (raynum / nrays_y) * (beam_max_x - beam_min_x) / (nrays_y - 1) + beam_min_x;
    y_init += dy/2;

    double ref = sqrt(square(x_init) + square(y_init));
    //if (ref > beam_max_x) return false;

    z_init = focal_length-dz/2;

    double theta1 = acos(beam_norm[beam*3+2]);
    double theta2 = atan2(beam_norm[beam*3+1]*focal_length, focal_length*beam_norm[beam*3+0]);

    double tmp_x = x_init;

    // ab: this might cause minor differences due to differences in CPU/GPU cos
    x_init = x_init*cos(theta1) + z_init*sin(theta1);	// first rotation
    z_init = z_init*cos(theta1) - tmp_x*sin(theta1);
    

    double tmp_x0 = x_init;
    x_init = x_init*cos(theta2) - y_init*sin(theta2);	// second rotation
    y_init = y_init*cos(theta2) + tmp_x0*sin(theta2);

    uray_init = (ref <= beam_max_x)*uray_mult*interp_cuda(pow_r, phase_r, ref, 2001);
    return true;
}

__global__
void launch_ray_XYZ(int b, unsigned nindices, double *eden, 
        double *etemp, double *edep, double *bbeam_norm,
        double *beam_norm, double *pow_r, double *phase_r,
        double xconst, double yconst, double zconst, int *marked, int *boxes,
        double *area_coverage, int *counter) {
    
    int beam = blockIdx.x + b*(nbeams/nGPUs);

    int start = blockIdx.y*blockDim.x + threadIdx.x;

    int lid = threadIdx.x % WARP_SIZE;
    int hlid = (lid % 2 == 0) ? lid + 1 : lid - 1;
    int vlid = ((lid)/rays_per_zone % 2 == 0) ? lid+rays_per_zone : lid-rays_per_zone;

    int search_index_x = 1, search_index_y = 1, search_index_z = 1,
        thisx_m, thisx_p, thisy_m, thisy_p, thisz_m, thisz_p;

    double dl, dm, dn, a1, a2, a3, a4, a5, a6, a7, a8, increment;
    double xtemp, ytemp, ztemp;
    double half = 0.5001;
    double myx, myy, myz, myvx, myvy, myvz, uray, uray_init;
    int thisx, thisy, thisz;

    /*__shared__ double ne_data[nr];
    __shared__ double r_data[nr];
    __shared__ double te_data[nr];

    __syncthreads();
    int rindices = ceil(nr/(float)threads_per_block);
    for (int i = 0; i < rindices; ++i) {
        int rindex = threadIdx.x + i*threads_per_block;
        if (rindex < nr) {
            ne_data[rindex] = ne_data_g[rindex];
            r_data[rindex] = r_data_g[rindex];
            te_data[rindex] = te_data_g[rindex];
        }
    }
    __syncthreads();*/

    int nthreads = min(max_threads, nrays*nbeams);
    int threads_per_beam = nthreads/nbeams;

    for (int r = 0; r < nindices; ++r) {
        int raynum = start + threads_per_beam*r;
        // raynum increases wrt to r so once this condition holds we are done 
        if (raynum >= nrays) return;
        bool t = init(beam, raynum, myx, myy, myz, uray, beam_norm, pow_r, phase_r);
        uray_init = uray;
        if (t) { 
        thisx = 0, thisy = 0, thisz = 0;
        for (int xx = 0; xx < nx; ++xx) {
            if (abs(xx*dx+xmin - myx) <= 0.5001 * dx) {
                thisx = xx;
                break;  // "breaks" out of the xx loop once the if statement condition is met.
            }
        }
        for (int yy = 0; yy < ny; ++yy) {
            if (abs(yy*dy+ymin - myy) <= 0.5001 * dy) {
                thisy = yy;
                break;  // "breaks" out of the yy loop once the if statement condition is met.
            }
        }
        for (int zz = 0; zz < nz; ++zz) {
            if (abs(zz*dz+zmin - myz) <= 0.5001 * dz) {
                thisz = zz;
                break;  // "breaks" out of the zz loop once the if statement condition is met.
            }
        }
        } else {
            continue;
        }
        // Calculate the total k (=sqrt(kx^2+kz^2)) from the dispersion relation,
        // taking into account the local plasma frequency of where the ray starts.
        //double wtmp = sqrt(square(thisx*dx+xmin) + square(thisy*dy+ymin) + square(thisz*dz+zmin));
        double wtmp = eden[eden_index(thisx, thisy, thisz)];
        double w = sqrt((square(omega) - wtmp*1e6*square(ec)/(me*e0)) / square(c));

        // Set the initial unnormalized k vectors, which give the initial direction
        // of the launched ray.
        // For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
        // For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.

        myvx = -1 * beam_norm[beam*3+0];
        myvy = -1 * beam_norm[beam*3+1];
        myvz = -1 * beam_norm[beam*3+2];

        // Length of k for the ray to be launched
        double knorm = sqrt(square(myvx) + square(myvy) + square(myvz));

        myvx = square(c) * ((myvx / knorm) * w) / omega;
        myvy = square(c) * ((myvy / knorm) * w) / omega;
        myvz = square(c) * ((myvz / knorm) * w) / omega;

        int numcrossing = 0;

        double lastx, lasty, lastz;
        int thisx_prev, thisy_prev, thisz_prev;

        // Time step loop
        for (int tt = 0; tt < nt; ++tt) {
            // The next ray position depends upon the discrete gradient
            // of a 3D array representing electron density
            // In order to avoid global memory accesses we compute both
            // the electron density and its gradient here
            thisx_m = thisx - search_index_x;
            thisx_p = thisx + search_index_x;
            thisy_m = thisy - search_index_y;
            thisy_p = thisy + search_index_y;
            thisz_m = thisz - search_index_z;
            thisz_p = thisz + search_index_z;
            if (thisx == 0) {
                thisx_p = 2;
                thisx_m = 0;
            } else if (thisx == nx-1) {
                thisx_p = nx-1;
                thisx_m = nx-3;
            }
            if (thisy == 0) {
                thisy_p = 2;
                thisy_m = 0;
            } else if (thisy == ny-1) {
                thisy_p = ny-1;
                thisy_m = ny-3;
            }
            if (thisz == 0) {
                thisz_p = 2;
                thisz_m = 0;
            } else if (thisz == nz-1) {
                thisz_p = nz-1;
                thisz_m = nz-3;
            }
            
            // Convert from coordinates in the grid
            // to coordinates in space and pow them
            /*double thisxp = thisx_p*dx+xmin;
            double thisxm = thisx_m*dx+xmin; 
            double thisxd = thisx*dx+xmin;
            double thisyp = thisy_p*dy+ymin; 
            double thisym = thisy_m*dy+ymin; 
            double thisyd = thisy*dy+ymin;
            double thiszp = thisz_p*dz+zmin; 
            double thiszm = thisz_m*dz+zmin; 
            double thiszd = thisz*dz+zmin;*/

            // Compute the electron density at each of the six directly
            // adjacent nodes
            /*double eden_x_p = interp_cuda(ne_data, r_data, 
                    sqrt(thisxp*thisxp + thisyd*thisyd + thiszd*thiszd),nr);
            double eden_x_m = interp_cuda(ne_data, r_data, 
                    sqrt(thisxm*thisxm + thisyd*thisyd + thiszd*thiszd),nr);
            double eden_y_p = interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyp*thisyp + thiszd*thiszd),nr);
            double eden_y_m = interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisym*thisym + thiszd*thiszd),nr);
            double eden_z_p = interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyd*thisyd + thiszp*thiszp),nr);
            double eden_z_m = interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyd*thisyd + thiszm*thiszm),nr);*/

			      double eden_x_p = eden[eden_index(thisx_p, thisy, thisz)];
			      double eden_x_m = eden[eden_index(thisx_m, thisy, thisz)];
			      double eden_y_p = eden[eden_index(thisx, thisy_p, thisz)];
			      double eden_y_m = eden[eden_index(thisx, thisy_m, thisz)];
			      double eden_z_p = eden[eden_index(thisx, thisy, thisz_p)];
			      double eden_z_m = eden[eden_index(thisx, thisy, thisz_m)];

            double old_x = myx;
            double old_y = myy;
            double old_z = myz;

            // Update ray position and velocity vectors
            myvx -= xconst * (eden_x_p - eden_x_m);
            myvy -= yconst * (eden_y_p - eden_y_m);
            myvz -= zconst * (eden_z_p - eden_z_m);
            myx += myvx * dt;
            myy += myvy * dt;
            myz += myvz * dt;

            // Compute ray coverage
            double h_x = __shfl_sync(0xffffffff, myx, hlid); 
            double v_x = __shfl_sync(0xffffffff, myx, vlid); 
            double h_y = __shfl_sync(0xffffffff, myy, hlid); 
            double v_y = __shfl_sync(0xffffffff, myy, vlid); 
            double h_z = __shfl_sync(0xffffffff, myz, hlid); 
            double v_z = __shfl_sync(0xffffffff, myz, vlid); 

            double x_hdiff = myx - h_x; 
            double y_hdiff = myy - h_y; 
            double z_hdiff = myz - h_z; 
            double x_vdiff = myx - v_x; 
            double y_vdiff = myy - v_y; 
            double z_vdiff = myz - v_z; 

            double area_x = y_hdiff*z_vdiff - z_hdiff*y_vdiff;
            double area_y = -x_hdiff*z_vdiff + z_hdiff*x_vdiff;
            double area_z = x_hdiff*y_vdiff - y_hdiff*z_vdiff;
    
            double area = sqrt(area_x*area_x + area_y*area_x + area_x*area_x)/2;
            
            // Helper values to simplify the following computations
            xtemp = (myx - xmin)*(1/dx);
            ytemp = (myy - ymin)*(1/dy);
            ztemp = (myz - zmin)*(1/dz);

            thisx_m = max(0,thisx-1);
            thisx_p = min(nx-1,thisx+1);
            thisy_m = max(0,thisy-1);
            thisy_p = min(ny-1,thisy+1);
            thisz_m = max(0,thisz-1);
            thisz_p = min(nz-1,thisz+1);

            // Determines current x index for the position
            // These loops count down to be consistent with the C++ code
            for (int xx = min(nx-1,thisx+1); xx >= max(0,thisx-1); --xx) {
                thisx = (abs(xx-xtemp) < half) ? xx : thisx;
            }
             // Determines current y index for the position
            for (int yy = min(ny-1,thisy+1); yy >= max(0,thisy-1); --yy) {
                thisy = (abs(yy-ytemp) < half) ? yy : thisy;
            }
            // Determines current z index for the position
            for (int zz = min(nz-1,thisz+1); zz >= max(0, thisz-1); --zz) {
                thisz = (abs(zz-ztemp) < half) ? zz : thisz;
            }

            // Code to update boxes
            // why do we need this loop?
            for (int xx = thisx_m; xx <= thisx_p; ++xx) {
              double currx = xx*dx+xmin - dx/2;
              if ((myx > currx && old_x <= currx) ||
                   myx < currx && old_x >= currx) {
                // can this still be 2D?
                double cross_z = old_z + (myz - old_z) / 
                                (myx - old_x) * (currx - old_x);
                double cross_y = old_y + (myy - old_y) / 
                                (myx - old_x) * (currx - old_x);
                // what does this condition do?
                if (abs(cross_z - lastz) > 1.0e-20) {
                  // don't store crossings that go out of bounds
                  if (myx <= xmax + dx/2 && myx >= xmin - dx/2) {
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 0] = thisx;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 1] = thisy;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 2] = thisz;
                    area_coverage[beam*nrays*ncrossings + raynum*ncrossings + numcrossing] = area;
                  } 
                  lastx = (int)currx;
                  numcrossing++;
                  break;
                }       
              }
            } 

            for (int yy = thisy_m; yy <= thisy_p; ++yy) {
              double curry = yy*dy+xmin - dy/2;
              if ((myy > curry && old_y <= curry) ||
                   myy < curry && old_y >= curry) {
                // can this still be 2D?
                double cross_z = old_z + (myz - old_z) / 
                                (myy - old_y) * (curry - old_y);
                double cross_x = old_x + (myx - old_x) / 
                                (myy - old_y) * (curry - old_y);
                // what does this condition do?
                if (abs(cross_x - lastx) > 1.0e-20) {
                  // don't store crossings that go out of bounds
                  if (myy <= ymax + dy/2 && myy >= ymin - dy/2) {
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 0] = thisx;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 1] = thisy;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 2] = thisz;
                    area_coverage[beam*nrays*ncrossings + raynum*ncrossings + numcrossing] = area;
                  } 
                  lasty = (int)curry;
                  numcrossing++;
                  break;
                }       
              }
            } 

            for (int zz = thisz_m; zz <= thisz_p; ++zz) {
              double currz = zz*dz+zmin - dz/2;
              if ((myz > currz && old_z <= currz) ||
                   myz < currz && old_z >= currz) {
                // can this still be 2D?
                double cross_y = old_y + (myy - old_y) / 
                                (myz - old_z) * (currz - old_z);
                double cross_x = old_x + (myx - old_x) / 
                                (myz - old_z) * (currz - old_z);
                // what does this condition do?
                if (abs(cross_x - lastx) > 1.0e-20) {
                  // don't store crossings that go out of bounds
                  if (myz <= zmax + dz/2 && myz >= zmin - dz/2) {
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 0] = thisx;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 1] = thisy;
                    boxes[beam*nrays*ncrossings*3 + raynum*ncrossings*3 + numcrossing*3 + 2] = thisz;
                    area_coverage[beam*nrays*ncrossings + raynum*ncrossings + numcrossing] = area;
                  } 
                  lastz = (int)currz;
                  numcrossing++;
                  break;
                }       
              }
            } 

            if (thisx != thisx_prev || thisy != thisy_prev || thisz != thisz_prev) {
              //if (beam*nrays+raynum == 195) printf("%d\n", thisx);
              int index = atomicAdd_system(&counter[eden_index(thisx, thisy, thisz)],1);
              atomicExch_system(&marked[thisx*ny*nz*1200 + thisy*nz*1200 + thisz*1200 + index], beam*nrays+raynum);
              //if (beam*nrays+raynum == 195) printf("%d\n", index);
            }
            thisx_prev = thisx;
            thisy_prev = thisy;
            thisz_prev = thisz;

            // In order to calculate the deposited energy into the plasma,
            // we need to calculate the plasma resistivity (eta) and collision frequency (nu_e-i)
			      double ed = eden[eden_index(thisx, thisy, thisz)];
			      double ete = etemp[eden_index(thisx, thisy, thisz)];
            double eta = 5.2e-5 * 10.0 / (ete*sqrt(ete));
            double nuei = (1e6 * ed * square(ec)/me)*eta;
        
            if (absorption == 1) {
                // Now we can decrement the ray's energy density according to how much energy
                // was absorbed by the plasma.
                increment = ed/ncrit * nuei * dt * uray;
                uray -= increment;
            } else {
                // We use this next line instead, if we are just using uray as a bookkeeping device
                // (i.e., no absorption by the plasma and no loss of energy by the ray).
                increment = uray;
            }

            // Rather than put all the energy into the cell in which the ray resides, which
            // is the so-called "nearest-neighbor" approach (which is very noise and less accurate),
            // we will use an area-based linear weighting scheme to deposit the energy to the
            // eight nearest nodes of the ray's current location. 

            // Define xp, yp and zp to be the ray's position relative to the nearest node.
            double xp = xtemp-thisx-0.5;
            double yp = ytemp-thisy-0.5;
            double zp = ztemp-thisz-0.5;

            // Below, we interpolate the energy deposition to the grid using linear area weighting.
            // The edep array must be two larger in each direction (one for min, one for max)
            // to accomodate this routine, since it deposits energy in adjacent cells.
            dm = 1.0 - abs(xp);
            dn = 1.0 - abs(yp);
            dl = 1.0 - abs(zp);
            a1 = (1.0-dl)*(1.0-dn)*(1.0-dm);
            a2 = (1.0-dl)*(1.0-dn)*dm;
            a3 = dl*(1.0-dn)*(1.0-dm);
            a4 = dl*(1.0-dn)*dm;
            a5 = (1.0-dl)*dn*(1.0-dm);
            a6 = (1.0-dl)*dn*dm;
            a7 = dl*dn*(1.0-dm);
            a8 = dl*dn*dm;

            int signx = ((xp < 0) ? -1 : 1), signy = ((yp < 0) ? -1 : 1),
                signz = ((zp < 0) ? -1 : 1);

            atomicAdd(&edep[edep_index(thisx+1, thisy+1, thisz+1)], a1*increment);
            atomicAdd(&edep[edep_index(thisx+1+signx, thisy+1, thisz+1)], a2*increment);
            atomicAdd(&edep[edep_index(thisx+1, thisy+1, thisz+1+signz)], a3*increment);
            atomicAdd(&edep[edep_index(thisx+1+signx, thisy+1, thisz+1+signz)], a4*increment);
            atomicAdd(&edep[edep_index(thisx+1, thisy+1+signy, thisz+1)], a5*increment);
            atomicAdd(&edep[edep_index(thisx+1+signx, thisy+1+signy, thisz+1)], a6*increment);
            atomicAdd(&edep[edep_index(thisx+1, thisy+1+signy, thisz+1+signz)], a7*increment);
            atomicAdd(&edep[edep_index(thisx+1+signx, thisy+1+signy, thisz+1+signz)], a8*increment);

            // This will cause the code to stop following the ray once it escapes the extent of the plasma
            if (uray <= 0.05 * uray_init ||
                myx < (xmin - (dx / 2.0)) || myx > (xmax + (dx / 2.0)) ||
                myy < (ymin - (dy / 2.0)) || myy > (ymax + (dy / 2.0)) ||
                myz < (zmin - (dz / 2.0)) || myz > (zmax + (dz / 2.0))) {
                break;
            }
        }
    }
}

