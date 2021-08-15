#include <iostream>
#include "def.cuh"
#include <cuda_runtime.h>
__device__
double interp_cuda(double *y, double *x, const double xp, int n)
{
    unsigned low, high, mid;
    //assert(x.size() == y.size());

    if (x[0] <= x[n-1]) {
        // x monotonically increase
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

        assert((xp >= x[mid]) && (xp <= x[mid + 1]));
        return y[mid] +
            (y[mid + 1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    } else {
        // x monotonically decrease
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

        assert((xp <= x[mid]) && (xp >= x[mid + 1]));
        return y[mid] +
            (y[mid +1] - y[mid]) / (x[mid + 1] - x[mid]) * (xp - x[mid]);
    }
}

__device__
bool init(int beam, int raynum, double &x_init, double &y_init, double &z_init, double &kx_init, double &ky_init, double &kz_init, double &uray_init, 
             double *beam_centers, const double *beam_norm, double *pow_r, double *phase_r) {
                 
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

    double ref = sqrt(pow(x_init, 2) + pow(y_init, 2));
    if (ref > beam_max_x) return false;
    z_init = focal_length-dz/2;
    double theta1 = acos(beam_centers[beam*3+2]/focal_length);
    double theta2 = atan2(beam_centers[beam*3+1], beam_centers[beam*3+0]);

    double tmp_x = x_init;

    // ab: this might cause minor differences due to differences in CPU/GPU cos
    x_init = x_init*cos(theta1) + z_init*sin(theta1);	// first rotation
    z_init = z_init*cos(theta1) - tmp_x*sin(theta1);
    

    double tmp_x0 = x_init;
    x_init = x_init*cos(theta2) - y_init*sin(theta2);	// second rotation
    y_init = y_init*cos(theta2) + tmp_x0*sin(theta2);

    kx_init = -1 * beam_norm[beam*3+0];
    ky_init = -1 * beam_norm[beam*3+1];
    kz_init = -1 * beam_norm[beam*3+2];

    uray_init = uray_mult*interp_cuda(pow_r, phase_r, ref, 2001);
    return true;
}

__global__
void launch_ray_XYZ(int beam, unsigned nindices,
                   double *x, double *y, double *z, double *wpe,
                   double *dedendx, double *dedendy, double *dedendz,
                   double *edep, double *eden, double *etemp, double *beam_centers, const double *beam_norm, double *pow_r, double *phase_r) {
    
    double x_init, y_init, z_init, kx_init, ky_init, kz_init, uray_init;
    beam = blockIdx.x;

    const int nt_d = ((1/courant_mult)*max(nx,nz)*2.0);
    int start = blockIdx.y*blockDim.x + threadIdx.x;

    for (int r = 0; r < nindices; ++r) {
        int raynum = start*nindices + r;
        if (raynum >= nrays) return;

        if (!init(beam, raynum, x_init, y_init, z_init, kx_init, ky_init, kz_init, uray_init, beam_centers, beam_norm, pow_r, phase_r)) continue;

        int search_index_x = 1, search_index_y = 1, search_index_z = 1,
            thisx_m, thisx_p, thisy_m, thisy_p, thisz_m, thisz_p;

        double dl, dm, dn, a1, a2, a3, a4, a5, a6, a7, a8, increment, nuei, eta;
        double myx, myy, myz, myvx, myvy, myvz, uray, uray0;
        uray = uray_init;
        myx = x_init;
        myy = y_init;
        myz = z_init;
        uray0 = uray;

        int thisx = 0, thisy = 0, thisz = 0;

        // ab: x[xx*ny*nz] is always equal to xx*dx+xmin
        for (int xx = 0; xx < nx; ++xx) {
            if (abs(myx - x[xx*ny*nz]) <= 0.5001 * dx) {
                thisx = xx;
                break;  // "breaks" out of the xx loop once the if statement condition is met.
            }
        }
        for (int yy = 0; yy < ny; ++yy) {
            if (abs(myy - y[yy*nz]) <= 0.5001 * dy) {
                thisy = yy;
                break;  // "breaks" out of the xx loop once the if statement condition is met.
            }
        }
        for (int zz = 0; zz < nz; ++zz) {
            if (abs(myz - z[zz]) <= 0.5001 * dz) {
                thisz = zz;
                break;  // "breaks" out of the zz loop once the if statement condition is met.
            }
        }

        /* Calculate the total k (=sqrt(kx^2+kz^2)) from the dispersion relation,
        * taking into account the local plasma frequency of where the ray starts.
        */
        double k = wpe[thisx*ny*nz+thisy*nz+thisz];

        /* Set the initial unnormalized k vectors, which give the initial direction
        * of the launched ray.
        * For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
        * For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.
        */

        // Length of k for the ray to be launched
        double knorm = sqrt(pow(kx_init, 2) + pow(ky_init, 2) + pow(kz_init, 2));

        // v_group, group velocity (dw/dk) from D(k,w)
        myvx = pow(c, 2) * ((kx_init / knorm) * k) / omega;
        myvy = pow(c, 2) * ((ky_init / knorm) * k) / omega;
        myvz = pow(c, 2) * ((kz_init / knorm) * k) / omega;

        // int thisx = 0, thisy = 0, thisz = 0;

        thisx_m = max(0, thisx - search_index_x);
        thisx_p = min(nx - 1, thisx + search_index_x);
        thisy_m = max(0, thisy - search_index_y);
        thisy_p = min(ny - 1, thisy + search_index_y);
        thisz_m = max(0, thisz - search_index_z);
        thisz_p = min(nz - 1, thisz + search_index_z);

        for (int xx = thisx_m; xx <= thisx_p; ++xx) {                  // Determines current x index for the position
            if (abs(x[xx*ny*nz] - myx) < 0.5001 * dx) {
                thisx = xx;
                break;  // "breaks" out of the xx loop once the if statement condition is met.
            }
        }
        
        for (int yy = thisy_m; yy <= thisy_p; ++yy) {                  // Determines current y index for the position
            if (abs(y[yy*nz] - myy) < 0.5001 * dy) {
                thisy = yy;
                break;  // "breaks" out of the yy loop once the if statement condition is met.
            }
        }

        for (int zz = thisz_m; zz <= thisz_p; ++zz) {                  // Determines current z index for the position
            if (abs(z[zz] - myz) < 0.5001 * dz) {
                thisz = zz;
                break;  // "breaks" out of the zz loop once the if statement condition is met.
            }
        }

        for (int tt = 1; tt < nt_d; ++tt) {                          // Time step loop
            // ==== Update the index, and track the intersections and boxes ====

            thisx_m = max(0, thisx - search_index_x);
            thisx_p = min(nx - 1, thisx + search_index_x);
            thisy_m = max(0, thisy - search_index_y);
            thisy_p = min(ny - 1, thisy + search_index_y);
            thisz_m = max(0, thisz - search_index_z);
            thisz_p = min(nz - 1, thisz + search_index_z);

            /* The (xi, yi, zi) logical indices for the position are now stored as
            (thisx, thisy, thisz). We also need to reset thisx_0, thisy_0, and
            thisz_0 to the new indices thisx, thisy, thisz.
            */

            myvz -= pow(c, 2) / (2.0 * ncrit) * dedendz[thisx*ny*nz+thisy*nz+thisz] * dt;
            myvy -= pow(c, 2) / (2.0 * ncrit) * dedendy[thisx*ny*nz+thisy*nz+thisz] * dt;
            myvx -= pow(c, 2) / (2.0 * ncrit) * dedendx[thisx*ny*nz+thisy*nz+thisz] * dt;
            myx += myvx * dt;
            myy += myvy * dt;
            myz += myvz * dt;

            // sz: Do we need this???
            for (int xx = thisx_m; xx <= thisx_p; ++xx) {                  // Determines current x index for the position
                if (abs(x[xx*ny*nz] - myx) < 0.5001 * dx) {
                    thisx = xx;
                    break;  // "breaks" out of the xx loop once the if statement condition is met.
                }
            }
            for (int yy = thisy_m; yy <= thisy_p; ++yy) {                  // Determines current y index for the position
                if (abs(y[yy*nz] - myy) < 0.5001 * dy) {
                    thisy = yy;
                    break;  // "breaks" out of the yy loop once the if statement condition is met.
                }
            }

            for (int zz = thisz_m; zz <= thisz_p; ++zz) {                  // Determines current z index for the position
                if (abs(z[zz] - myz) < 0.5001 * dz) {
                    thisz = zz;
                    break;  // "breaks" out of the zz loop once the if statement condition is met.
                }
            }

            /* The (xi, yi, zi) logical indices for the position are now stored as
            (thisx, thisy, thisz). We also need to reset thisx_0, thisy_0, and
            thisz_0 to the new indices thisx, thisy, thisz.
            */

            /* In order to calculate the deposited energy into the plasma,
            we need to calculate the plasma resistivity (eta) and collision frequency (nu_e-i) */

            /* Now we can decrement the ray's energy density according to how much energy
            was absorbed by the plasma. */

            /* We use these next two lines instead, if we are just using uray as a bookkeeping device
                (i.e., no absorption by the plasma and no loss of energy by the ray).	*/
            //TODO Move comments
            eta = etemp[thisx*ny*nz+thisy*nz+thisz]; // ohm*m
            nuei = (1e6 * eden[thisx*nx*ny+thisy*nz+thisz] * pow(ec, 2)/me) * eta;
        
            if (absorption == 1) {
                /* Now we can decrement the ray's energy density according to how much energy
                    was absorbed by the plasma. */
                increment = nuei * (eden[thisx*ny*nz+thisy*nz+thisz]/ncrit) * uray*dt;
                uray -= increment;
                //increment = nuei * (eden[thisx*ny*nz+thisy*nz+thisz]/ncrit) * uray*dt;
            } else {
                /* We use these next two lines instead, if we are just using uray as a bookkeeping device
                    (i.e., no absorption by the plasma and no loss of energy by the ray).	*/
                //uray = uray;
                increment = uray;
            }

            /* Rather than put all the energy into the cell in which the ray resides, which
            is the so-called "nearest-neighbor" approach (which is very noise and less accurate),
            we will use an area-based linear weighting scheme to deposit the energy to the
            four nearest nodes of the ray's current location. In 3D, this would be 8 nearest.   */

            // Define xp and zp to be the ray's position relative to the nearest node.
            double xp = (myx - (x[thisx*ny*nz+thisy*nz+thisz] + dx / 2.0)) / dx;
            double yp = (myy - (y[thisx*ny*nz+thisy*nz+thisz] + dy / 2.0)) / dy;
            double zp = (myz - (z[thisx*ny*nz+thisy*nz+thisz] + dz / 2.0)) / dz;

    /*	Below, we interpolate the energy deposition to the grid using linear area weighting.
        The edep array must be two larger in each direction (one for min, one for max)
        to accomodate this routine, since it deposits energy in adjacent cells.		*/

            dl = abs(zp);
            dm = abs(xp);
            dn = abs(yp);

            a1 = dl*dn*dm;	// blue		: (x  , y, z  )
            a2 = dl*dn*(1.0-dm);		// green	: (x+1, y, z  )
            a3 = (1.0-dl)*dn*dm;		// yellow	: (x  , y, z+1)
            a4 = (1.0-dl)*dn*(1.0-dm);			// red 		: (x+1, y, z+1)
            a5 = dl*(1.0-dn)*dm;	// blue		: (x  , y+1, z  )
            a6 = dl*(1.0-dn)*(1.0-dm);		// green	: (x+1, y+1, z  )
            a7 = (1.0-dl)*(1.0-dn)*dm;		// yellow	: (x  , y+1, z+1)
            a8 = (1.0-dl)*(1.0-dn)*(1.0-dm);			// red 		: (x+1, y+1, z+1)

            int signx = ((xp < 0) ? -1 : 1), signy = ((yp < 0) ? -1 : 1),
                signz = ((zp < 0) ? -1 : 1);


            atomicAdd(&edep[(thisx + 1)*(ny+2)*(nz+2)+(thisy + 1)*(nz+2)+thisz + 1], a1 * increment);
            atomicAdd(&edep[(thisx + 1 + signx)*(ny+2)*(nz+2)+(thisy + 1)*(nz+2)+thisz + 1], a2 * increment);
            atomicAdd(&edep[(thisx + 1)*(ny+2)*(nz+2)+(thisy + 1)*(nz+2)+thisz + 1 + signz], a3 * increment);
            atomicAdd(&edep[(thisx + 1 + signx)*(ny+2)*(nz+2)+(thisy + 1)*(nz+2)+thisz + 1 + signz], a4 * increment);
            atomicAdd(&edep[(thisx + 1)*(ny+2)*(nz+2)+(thisy + 1 + signy)*(nz+2)+thisz + 1], a5 * increment);
            atomicAdd(&edep[(thisx + 1 + signx)*(ny+2)*(nz+2)+(thisy + 1 + signy)*(nz+2)+thisz + 1], a6 * increment);
            atomicAdd(&edep[(thisx + 1)*(ny+2)*(nz+2)+(thisy + 1 + signy)*(nz+2)+thisz + 1 + signz], a7 * increment);
            atomicAdd(&edep[(thisx + 1 + signx)*(ny+2)*(nz+2)+(thisy + 1 + signy)*(nz+2)+thisz + 1 + signz], a8 * increment);


            /* This is how the amplitude of the E field is changed during propagation, but is not derived here.*/
            /* It assumes conservation of wave action. */
            // This will cause the code to stop following the ray once it escapes the extent of the plasma
            if (uray <= 0.05 * uray0 ||
                myx < (xmin - (dx / 2.0)) || myx > (xmax + (dx / 2.0)) ||
                myy < (ymin - (dy / 2.0)) || myy > (ymax + (dy / 2.0)) ||
                myz < (zmin - (dz / 2.0)) || myz > (zmax + (dz / 2.0))) {
                break;                  // "breaks" out of the tt loop once the if condition is satisfied
            }
        }
    }
}

