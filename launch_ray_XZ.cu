#include <iostream>
#include "def.cuh"
#include <cuda_runtime.h>

__device__ long edep_index(long x, long y, long z) {
    return x*(ny+2)*(nz+2) + y*(nz+2) + z;
}

__device__ position_type pos_square(position_type x){
    return x*x;
}

__device__ energy_type en_square(energy_type x){
    return x*x;
}

// Piecewise linear interpolation
// Use binary search to find the segment
// Ref: https://software.llnl.gov/yorick-doc/qref/qrfunc09.html
__device__ position_type pos_interp_cuda(position_type *y, position_type *x, const position_type xp, int n)
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

__device__ energy_type en_interp_cuda(energy_type *y, energy_type *x, const energy_type xp, int n)
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
bool init(int beam, int pre_raynum, position_type &x_init, position_type &y_init, position_type &z_init,
    energy_type &uray_init, const position_type *beam_norm, energy_type *pow_r, energy_type *phase_r) {

    int zones_spanned = ceil((beam_max_x-beam_min_x)/xres);
    int b1 = pre_raynum/(rays_per_zone*rays_per_zone);
    int b2 = pre_raynum%(rays_per_zone*rays_per_zone);
    int ry = b1/(zones_spanned)*rays_per_zone + b2/rays_per_zone;
    int rx = b1%(zones_spanned)*rays_per_zone + b2%rays_per_zone;
    int raynum = ry*nrays_x+rx;

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

    position_type ref = sqrt(pos_square(x_init) + pos_square(y_init));
    //if (ref > beam_max_x) return false;

    z_init = focal_length-dz/2;

    // position_type theta1 = acos(beam_norm[beam*3+2]);
    // position_type theta2 = atan2(beam_norm[beam*3+1]*focal_length, focal_length*beam_norm[beam*3+0]);
    // if (beam*nrays+pre_raynum == 5000) printf("%f %f\n", theta1, theta2);
    // if (beam*nrays+pre_raynum == 5000) printf("%lf %lf\n", acos(beam_norm[beam*3+2]), atan2(beam_norm[beam*3+1]*focal_length, focal_length*beam_norm[beam*3+0]));

    position_type tmp_x = x_init;

    // // ab: this might cause minor differences due to differences in CPU/GPU cos
    // x_init = x_init*cos(theta1) + z_init*sin(theta1);	// first rotation
    // z_init = z_init*cos(theta1) - tmp_x*sin(theta1);

    x_init = x_init*beam_norm[beam*4] + z_init*beam_norm[beam*4+1];	// first rotation
    z_init = z_init*beam_norm[beam*4] - tmp_x*beam_norm[beam*4+1];
    

    // position_type tmp_x0 = x_init;
    // x_init = x_init*cos(theta2) - y_init*sin(theta2);	// second rotation
    // y_init = y_init*cos(theta2) + tmp_x0*sin(theta2);

    position_type tmp_x0 = x_init;
    x_init = x_init*beam_norm[beam*4+2] - y_init*beam_norm[beam*4+3];	// second rotation
    y_init = y_init*beam_norm[beam*4+2] + tmp_x0*beam_norm[beam*4+3];

    uray_init = uray_mult*en_interp_cuda(pow_r, phase_r, (energy_type)ref, 2001);
    return ref <= beam_max_x;
}

__global__
void launch_ray_XYZ(int b, unsigned nindices, energy_type *te_data_g, 
    energy_type *r_data_g, energy_type *ne_data_g, energy_type *edep, position_type *bbeam_norm,
    position_type *beam_norm, energy_type *pow_r, energy_type *phase_r,
    position_type xconst, position_type yconst, position_type zconst) {
    
    int beam = blockIdx.x + b*(nbeams/nGPUs);

    int start = blockIdx.y*blockDim.x + threadIdx.x;

    int search_index_x = 1, search_index_y = 1, search_index_z = 1,
        thisx_m, thisx_p, thisy_m, thisy_p, thisz_m, thisz_p;

    energy_type dl, dm, dn, a1, a2, a3, a4, a5, a6, a7, a8, increment;
    position_type xtemp, ytemp, ztemp;
    position_type myx, myy, myz, myvx, myvy, myvz;
    energy_type uray, uray_init;
    int thisx, thisy, thisz;

    __shared__ position_type ne_data[nr];
    __shared__ position_type r_data[nr];
    __shared__ position_type te_data[nr];

    __syncthreads();
    int rindices = ceil(nr/(float)threads_per_block);
    for (int i = 0; i < rindices; ++i) {
        int rindex = threadIdx.x + i*threads_per_block;
        if (rindex < nr) {
            ne_data[rindex] = (position_type)ne_data_g[rindex];
            r_data[rindex] = (position_type)r_data_g[rindex];
            te_data[rindex] = te_data_g[rindex];
        }
    }
    __syncthreads();

    int nthreads = min(max_threads, nrays*nbeams);
    int threads_per_beam = nthreads/nbeams;

    for (int r = 0; r < nindices; ++r) {
        int raynum = start + threads_per_beam*r;
        // raynum increases wrt to r so once this condition holds we are done 
        if (raynum >= nrays) return;
        bool t = init(beam, raynum, myx, myy, myz, uray, bbeam_norm, pow_r, phase_r);
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
        position_type wtmp = sqrt(pos_square(thisx*dx+xmin) + pos_square(thisy*dy+ymin) + pos_square(thisz*dz+zmin));
        wtmp = pos_interp_cuda(ne_data, r_data, wtmp, nr);
        position_type w = sqrt((pos_square(omega) - wtmp*1e6*pos_square(ec)/((double)me*e0)) / pos_square(c));
        // if (beam*nrays+raynum == 5000) printf("%.40f, %d\n", pos_square(ec), pos_square(ec) == 0.0);

        // Set the initial unnormalized k vectors, which give the initial direction
        // of the launched ray.
        // For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
        // For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.

        myvx = -1 * beam_norm[beam*3+0];
        myvy = -1 * beam_norm[beam*3+1];
        myvz = -1 * beam_norm[beam*3+2];

        // Length of k for the ray to be launched
        position_type knorm = sqrt(pos_square(myvx) + pos_square(myvy) + pos_square(myvz));

        myvx = pos_square(c) * ((myvx / knorm) * w) / omega;
        myvy = pos_square(c) * ((myvy / knorm) * w) / omega;
        myvz = pos_square(c) * ((myvz / knorm) * w) / omega;

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
            position_type thisxp = thisx_p*dx+xmin;
            position_type thisxm = thisx_m*dx+xmin; 
            position_type thisxd = thisx*dx+xmin;
            position_type thisyp = thisy_p*dy+ymin; 
            position_type thisym = thisy_m*dy+ymin; 
            position_type thisyd = thisy*dy+ymin;
            position_type thiszp = thisz_p*dz+zmin; 
            position_type thiszm = thisz_m*dz+zmin; 
            position_type thiszd = thisz*dz+zmin;

            // Compute the electron density at each of the six directly
            // adjacent nodes
            position_type eden_x_p = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxp*thisxp + thisyd*thisyd + thiszd*thiszd),nr);
            position_type eden_x_m = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxm*thisxm + thisyd*thisyd + thiszd*thiszd),nr);
            position_type eden_y_p = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyp*thisyp + thiszd*thiszd),nr);
            position_type eden_y_m = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisym*thisym + thiszd*thiszd),nr);
            position_type eden_z_p = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyd*thisyd + thiszp*thiszp),nr);
            position_type eden_z_m = pos_interp_cuda(ne_data, r_data, 
                    sqrt(thisxd*thisxd + thisyd*thisyd + thiszm*thiszm),nr);

            // Update ray position and velocity vectors
            myvx -= xconst * (eden_x_p - eden_x_m);
            myvy -= yconst * (eden_y_p - eden_y_m);
            myvz -= zconst * (eden_z_p - eden_z_m);
            myx += myvx * dt;
            myy += myvy * dt;
            myz += myvz * dt;

            // if (beam*nrays+raynum == 5000) printf("%lf %lf %lf\n", xconst * (eden_x_p - eden_x_m), eden_x_p, sqrt(thisxp*thisxp + thisyd*thisyd + thiszd*thiszd));
            
            // Helper values to simplify the following computations
            xtemp = (myx - xmin)*(1/dx);
            ytemp = (myy - ymin)*(1/dy);
            ztemp = (myz - zmin)*(1/dz);

            // Determines current x index for the position
            // These loops count down to be consistent with the C++ code
            for (int xx = min(nx-1,thisx+1); xx >= max(0,thisx-1); --xx) {
                thisx = (abs(xx-xtemp) < 0.5001) ? xx : thisx;
            }
             // Determines current y index for the position
            for (int yy = min(ny-1,thisy+1); yy >= max(0,thisy-1); --yy) {
                thisy = (abs(yy-ytemp) < 0.5001) ? yy : thisy;
            }
            // Determines current z index for the position
            for (int zz = min(nz-1,thisz+1); zz >= max(0, thisz-1); --zz) {
                thisz = (abs(zz-ztemp) < 0.5001) ? zz : thisz;
            }

            // if (beam*nrays+raynum == 5000) printf("%d %d %d\n", thisx, thisy, thisz);

            // In order to calculate the deposited energy into the plasma,
            // we need to calculate the plasma resistivity (eta) and collision frequency (nu_e-i)
            energy_type tmp = sqrt(en_square(thisx*dx+xmin) + en_square(thisy*dy+ymin) + en_square(thisz*dz+zmin));
            energy_type ed = en_interp_cuda(ne_data_g, r_data_g, tmp, nr);
			energy_type etemp = en_interp_cuda(te_data_g, r_data_g, tmp, nr);
            energy_type eta = 5.2e-5 * 10.0 / (etemp*sqrt(etemp));
            energy_type nuei = (1e6 * ed * en_square(ec)/me)*eta;
        
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
            energy_type xp = xtemp-thisx-0.5;
            energy_type yp = ytemp-thisy-0.5;
            energy_type zp = ztemp-thisz-0.5;

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

