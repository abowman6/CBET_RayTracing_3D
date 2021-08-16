#include <iostream>
#include "def.cuh"
#include <cuda_runtime.h>

__device__
bool isInChunk(int thisx, int thisy, int thisz, int i, int j, int k) {
    return thisx >= i && thisy >= j && thisz >= k
           && thisx < i + chunk_size && thisy < j + chunk_size && thisz < k + chunk_size;
}

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
bool init(int beam, int raynum, double &x_init, double &y_init, double &z_init,
            double &uray_init, const double *beam_norm, double *pow_r, double *phase_r) {
                 
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

    double theta1 = acos(beam_norm[beam*3+2]);
    double theta2 = atan2(beam_norm[beam*3+1]*focal_length, focal_length*beam_norm[beam*3+0]);

    double tmp_x = x_init;

    // ab: this might cause minor differences due to differences in CPU/GPU cos
    x_init = x_init*cos(theta1) + z_init*sin(theta1);	// first rotation
    z_init = z_init*cos(theta1) - tmp_x*sin(theta1);
    

    double tmp_x0 = x_init;
    x_init = x_init*cos(theta2) - y_init*sin(theta2);	// second rotation
    y_init = y_init*cos(theta2) + tmp_x0*sin(theta2);

    uray_init = uray_mult*interp_cuda(pow_r, phase_r, ref, 2001);
    return true;
}

__global__
void calculate_myxyz(int nindices, double *beam_norm, double *pow_r,
                    double *phase_r, double *uray_arr, double *uray_i, int *time_passed,
                    int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
                    double *x, double *y, double *z, int *counter,
                    double *myx_arr, double *myy_arr, double *myz_arr, double uray_mult_true, double focal_length_true) {
    
    int beam = blockIdx.x;
    double myx, myy, myz, uray;
    int search_index_x = 1, search_index_y = 1, search_index_z = 1,
        thisx_m, thisx_p, thisy_m, thisy_p, thisz_m, thisz_p;
    for (int i = 0; i < nindices; ++i) {
        int raynum = (blockIdx.y*blockDim.x + threadIdx.x)*nindices + i;
        
        if (raynum >= nrays) return;
        bool t = init(beam, raynum, myx, myy, myz, uray, beam_norm, pow_r, phase_r);
        if (t) {

            myx_arr[beam*nrays+raynum] = myx;
            myy_arr[beam*nrays+raynum] = myy;
            myz_arr[beam*nrays+raynum] = myz;
            uray_arr[beam*nrays+raynum] = uray;
            uray_i[beam*nrays+raynum] = uray;

            int thisx_0 = 0, thisy_0 = 0, thisz_0 = 0;

            for (int xx = 0; xx < nx; ++xx) {
                if (abs(xx*dx+xmin - myx) <= 0.5001 * dx) {
                    thisx_0 = xx;
                    break;  // "breaks" out of the xx loop once the if statement condition is met.
                }
            }
            for (int yy = 0; yy < ny; ++yy) { //yy*dy*ymin
                if (abs(yy*dy+ymin - myy) <= 0.5001 * dy) {
                    thisy_0 = yy;
                    break;  // "breaks" out of the yy loop once the if statement condition is met.
                }
            }
            for (int zz = 0; zz < nz; ++zz) { //zz*dz*zmin
                if (abs(zz*dz+zmin - myz) <= 0.5001 * dz) {
                    thisz_0 = zz;
                    break;  // "breaks" out of the zz loop once the if statement condition is met.
                }
            }

            thisx_m = max(0, thisx_0 - search_index_x);
            thisx_p = min(nx - 1, thisx_0 + search_index_x);
            thisy_m = max(0, thisy_0 - search_index_y);
            thisy_p = min(ny - 1, thisy_0 + search_index_y);
            thisz_m = max(0, thisz_0 - search_index_z);
            thisz_p = min(nz - 1, thisz_0 + search_index_z);

            for (int xx = thisx_m; xx <= thisx_p; ++xx) {                  // Determines current x index for the position
                if (abs(xx*dx+xmin - myx) < 0.5001 * dx) {
                    thisx_0 = xx;
                    break;  // "breaks" out of the xx loop once the if statement condition is met.
                }
            }
            
            for (int yy = thisy_m; yy <= thisy_p; ++yy) {                  // Determines current y index for the position
                if (abs(yy*dy+ymin - myy) < 0.5001 * dy) {
                    thisy_0 = yy;
                    break;  // "breaks" out of the yy loop once the if statement condition is met.
                }
            }

            for (int zz = thisz_m; zz <= thisz_p; ++zz) {                  // Determines current z index for the position
                if (abs(zz*dz+zmin - myz) < 0.5001 * dz) {
                    thisz_0 = zz;
                    break;  // "breaks" out of the zz loop once the if statement condition is met.
                }
            }

            thisx_0_arr[beam*nrays+raynum] = thisx_0;
            thisy_0_arr[beam*nrays+raynum] = thisy_0;
            thisz_0_arr[beam*nrays+raynum] = thisz_0;
            
        } else {
            atomicAdd(counter, 1);
            time_passed[beam*nrays+raynum] = -1;
        }
    }
}

__global__
void initial_launch(int nindices, double *wpe, double *beam_norm, double *uray_arr,
                    int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
                    double *myvx_arr, double *myvy_arr, double *myvz_arr, int *time_passed,
                    int i, int j, int k) {
    
    double kx_init, ky_init, kz_init;
    int beam = blockIdx.x;
    
    for (int m = 0; m < nindices; ++m) {
        int raynum = (blockIdx.y*blockDim.x + threadIdx.x)*nindices + m;
        if (raynum >= nrays) return;
        if (time_passed[beam*nrays+raynum] == -1) continue;

        int thisx_0 = thisx_0_arr[beam*nrays+raynum];
        int thisy_0 = thisy_0_arr[beam*nrays+raynum];
        int thisz_0 = thisz_0_arr[beam*nrays+raynum];

        if (!isInChunk(thisx_0, thisy_0, thisz_0, i, j, k)) continue; 
        int thisxl = thisx_0 % chunk_size;
        int thisyl = thisy_0 % chunk_size;
        int thiszl = thisz_0 % chunk_size;
        /* Calculate the total k (=sqrt(kx^2+kz^2)) from the dispersion relation,
        * taking into account the local plasma frequency of where the ray starts. */
        double k1 = wpe[thisxl*chunk_size*chunk_size+thisyl*chunk_size+thiszl];
        
        //if (beam*nrays+raynum == 256517) printf("test %lf %lf\n", k1, wpe[thisxl*chunk_size*chunk_size+thisyl*chunk_size+thiszl]);
        //if (beam*nrays+raynum == 256517) printf("%d\n", thisxl*chunk_size*chunk_size+thisyl*chunk_size+thiszl);
        //if (beam*nrays+raynum == 256517) printf("%d %d %d\n", i, j, k);
        //if (beam*nrays+raynum == 256517) { 
            //for (int xx = 0; xx < c3; ++xx) {
               //if (xx == thisxl*chunk_size*chunk_size+thisyl*chunk_size+thiszl) printf("%lf\n", wpe[xx]);
            //}
            //printf("test %lf\n", wpe[thisxl*chunk_size*chunk_size+thisyl*chunk_size+thiszl]);
            //printf("test %lf\n", k1);
        //}
        // double k = 1;

        /* Set the initial unnormalized k vectors, which give the initial direction
        * of the launched ray.
        * For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
        * For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.
        */

        kx_init = -1 * beam_norm[beam*3+0];
        ky_init = -1 * beam_norm[beam*3+1];
        kz_init = -1 * beam_norm[beam*3+2];
        //if(beam*nrays+raynum == 256517) printf("%lf\n", kz_init);
        // Length of k for the ray to be launched
        double knorm = sqrt(pow(kx_init, 2) + pow(ky_init, 2) + pow(kz_init, 2));
        // ab: we should be able to move this into launch ray and save the 
        // chunking effort for wpe.
        myvx_arr[beam*nrays+raynum] = pow(c, 2) * ((kx_init / knorm) * k1) / omega;
        myvy_arr[beam*nrays+raynum] = pow(c, 2) * ((ky_init / knorm) * k1) / omega;
        myvz_arr[beam*nrays+raynum] = pow(c, 2) * ((kz_init / knorm) * k1) / omega;
    }
}

__global__
void launch_ray_XYZ(int b, unsigned nindices,
                   double *x, double *y, double *z,
                   double *dedendx, double *dedendy, double *dedendz,
                   double *edep, double *eden, double *etemp, double *uray_arr,
                   double *myx_arr, double *myy_arr, double *myz_arr,
                   double *myvx_arr, double *myvy_arr, double *myvz_arr,
                   double *uray_init,
                   int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
                   int i, int j, int k, int *time_passed, int *update, int *counter) {
    
    int beam = blockIdx.x + b*nbeams/2;

    const int nt_d = ((1/courant_mult)*max(nx,nz)*2.0);

    int start = blockIdx.y*blockDim.x + threadIdx.x;
    // double temp = ncrit;

    int search_index_x = 1, search_index_y = 1, search_index_z = 1,
        thisx_m, thisx_p, thisy_m, thisy_p, thisz_m, thisz_p;

    double dl, dm, dn, a1, a2, a3, a4, a5, a6, a7, a8, increment, nuei, eta;

    for (int r = 0; r < nindices; ++r) {
        int raynum = start*nindices + r;
        // ab: raynum increases wrt to r so one this condition holds we are done 
        if (raynum >= nrays) return;
        //if (beam*nrays+raynum == 51) printf("%d\n", b);
       
        int thisx = thisx_0_arr[beam*nrays+raynum];
        int thisy = thisy_0_arr[beam*nrays+raynum];
        int thisz = thisz_0_arr[beam*nrays+raynum];

        if (!isInChunk(thisx, thisy, thisz, i, j, k)) continue;

        int tt = time_passed[beam*nrays+raynum];
        if (tt == -1) continue;

        double myx = myx_arr[beam*nrays+raynum];
        double myy = myy_arr[beam*nrays+raynum];
        double myz = myz_arr[beam*nrays+raynum];

        double myvx = myvx_arr[beam*nrays+raynum];
        double myvy = myvy_arr[beam*nrays+raynum];
        double myvz = myvz_arr[beam*nrays+raynum];
        double uray = uray_arr[beam*nrays+raynum];
        double uray0 = uray_init[beam*nrays+raynum];
 

        //if (beam*nrays+raynum == 256517) printf("start %lf %lf %lf\n", myx, myy, myz);
    
        //int thisx = thisx_0, thisy = thisy_0, thisz = thisz_0;


        for (; tt < nt_d; ++tt) {                          // Time step loop
            // ==== Update the index, and track the intersections and boxes ====

            thisx_m = max(0, thisx - search_index_x);
            thisx_p = min(nx - 1, thisx + search_index_x);
            thisy_m = max(0, thisy - search_index_y);
            thisy_p = min(ny - 1, thisy + search_index_y);
            thisz_m = max(0, thisz - search_index_z);
            thisz_p = min(nz - 1, thisz + search_index_z);

            int thisx_l = thisx % chunk_size;
            int thisy_l = thisy % chunk_size;
            int thisz_l = thisz % chunk_size;
            
            // make sure that the values are updated once per time step
            if (!update[beam*nrays+raynum]) {
                //if (beam*nrays+raynum == 256517) printf("%d %d %d %lf\n", thisx, thisy, thisz, myvz);
/*
                myvz -= pow(c, 2) / (2.0 * ncrit) * 
                    dedendz[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l] * dt;
                myvy -= pow(c, 2) / (2.0 * ncrit) * 
                    dedendy[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l] * dt;
                myvx -= pow(c, 2) / (2.0 * ncrit) * 
                    dedendx[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l] * dt;
*/              myvz -= 
                    dedendz[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l];
                    //1;
                myvy -= 
                    dedendy[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l];
                    //1;
                myvx -= 
                    dedendx[thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l];
                    //1;
                myx += myvx * dt;
                myy += myvy * dt;
                myz += myvz * dt;
            } else {
                update[beam*nrays+raynum] = 0;
            }

            for (int xx = thisx_m; xx <= thisx_p; ++xx) {                  // Determines current x index for the position
                if (abs(xx*dx+xmin - myx) < 0.5001 * dx) {
                    thisx = xx;
                    break;  // "breaks" out of the xx loop once the if statement condition is met.
                }
            }
            for (int yy = thisy_m; yy <= thisy_p; ++yy) {                  // Determines current y index for the position
                if (abs(yy*dy+ymin - myy) < 0.5001 * dy) {
                    thisy = yy;
                    break;  // "breaks" out of the yy loop once the if statement condition is met.
                }
            }

            for (int zz = thisz_m; zz <= thisz_p; ++zz) {                  // Determines current z index for the position
                if (abs(zz*dz+zmin - myz) < 0.5001 * dz) {
                    thisz = zz;
                    break;  // "breaks" out of the zz loop once the if statement condition is met.
                }
            }
            if (!isInChunk(thisx, thisy, thisz, i, j, k)) {
                update[beam*nrays+raynum] = 1;
                break;
            }
            //if (beam*nrays+raynum == 51) printf("%d %d %d\n", thisx, thisy, thisz);
            //if (beam*nrays+raynum == 41427) printf("%lf\n", myz);
            
            thisx_l = thisx % chunk_size;
            thisy_l = thisy % chunk_size;
            thisz_l = thisz % chunk_size;

            //assert(thisx_l+i==thisx);
            //assert(thisy_l+j==thisy);
            //assert(thisz_l+k==thisz);

            int index = thisx_l*chunk_size*chunk_size+thisy_l*chunk_size+thisz_l;

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
        
            // eta = etemp[index]; // ohm*m
            // nuei = (1e6 * eden[index] * pow(ec, 2)/me) * eta;
        
            if (absorption == 1) {
                /* Now we can decrement the ray's energy density according to how much energy
                    was absorbed by the plasma. */
                increment = eden[index] * uray;
                //increment = 7.5423 * uray;
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
            four nearest nodes of the ray's current location. In 3D, this is the 8 nearest.   */

            // Define xp and zp to be the ray's position relative to the nearest node.
            double xp = (myx - (thisx*dx+xmin + dx / 2.0)) / dx;
            double yp = (myy - (thisy*dy+ymin + dx / 2.0)) / dx;
            double zp = (myz - (thisz*dz+zmin + dx / 2.0)) / dx;

        /*	Below, we interpolate the energy deposition to the grid using linear area weighting.
            The edep array must be two larger in each direction (one for min, one for max)
            to accomodate this routine, since it deposits energy in adjacent cells.		*/

            dl = 1.0 - abs(zp);
            dm = 1.0 - abs(xp);
            dn = 1.0 - abs(yp);
            a1 = (1.0-dl)*(1.0-dn)*(1.0-dm);	// blue		: (x  , y, z  )
            a2 = (1.0-dl)*(1.0-dn)*dm;		// green	: (x+1, y, z  )
            a3 = dl*(1.0-dn)*(1.0-dm);		// yellow	: (x  , y, z+1)
            a4 = dl*(1.0-dn)*dm;			// red 		: (x+1, y, z+1)
            a5 = (1.0-dl)*dn*(1.0-dm);	// blue		: (x  , y+1, z  )
            a6 = (1.0-dl)*dn*dm;		// green	: (x+1, y+1, z  )
            a7 = dl*dn*(1.0-dm);		// yellow	: (x  , y+1, z+1)
            a8 = dl*dn*dm;			// red 		: (x+1, y+1, z+1)			// red 		: (x+1, y+1, z+1)

            int signx = ((xp < 1e-13) ? -1 : 1), signy = ((yp < 1e-13) ? -1 : 1),
                signz = ((zp < 1e-13) ? -1 : 1);   
#if 0
            int x1 = 0; int y1 = 0; int z1 = 0; 
            if (thisx+1 == x1 && thisy+1 == y1 && thisz+1+signz == z1) printf("1 %d\n", beam*nrays+raynum);
            if (thisx+1 == x1 && thisy+1 == y1 && thisz+1 == z1) printf("2 %d\n", beam*nrays+raynum);
            if (thisx+1 == x1 && thisy+1+signy == y1 && thisz+1+signz == z1) printf("3 %d\n", beam*nrays+raynum);
            if (thisx+1 == x1 && thisy+1+signy == y1 && thisz+1 == z1) printf("4 %d\n", beam*nrays+raynum);
            if (thisx+1+signx == x1 && thisy+1 == y1 && thisz+1+signz == z1) printf("5 %d\n", beam*nrays+raynum);
            if (thisx+1+signx == x1 && thisy+1 == y1 && thisz+1 == z1) printf("6 %d\n", beam*nrays+raynum);
            if (thisx+1+signx == x1 && thisy+1+signy == y1 && thisz+1+signz == z1) printf("7 %d\n", beam*nrays+raynum);
            if (thisx+1+signx == x1 && thisy+1+signy == y1 && thisz+1 == z1) printf("8 %d\n", beam*nrays+raynum);
#endif
            //x[thisx*nz*ny*nbeams*nrays+thisy*nz*nbeams*nrays+thisz*nbeams*nrays+beam*nrays+raynum] = ;
#if 1
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1], a1*increment);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1], a2*increment);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1 + signz], a3*increment);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1 + signz], a4*increment);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1], a5*increment);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1], a6*increment);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1 + signz], a7*increment);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1 + signz], a8*increment);
#else
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1], 1);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1], 1);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1 + signz], 1);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1)*(chunk_size+2)+thisz_l + 1 + signz], 1);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1], 1);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1], 1);
            atomicAdd(&edep[(thisx_l + 1)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1 + signz], 1);
            atomicAdd(&edep[(thisx_l + 1 + signx)*(chunk_size+2)*(chunk_size+2)+(thisy_l + 1 + signy)*(chunk_size+2)+thisz_l + 1 + signz], 1);
#endif
            /* This is how the amplitude of the E field is changed during propagation, but is not derived here.*/
            /* It assumes conservation of wave action. */
            // This will cause the code to stop following the ray once it escapes the extent of the plasma
            if (uray <= 0.05 * uray0 ||
                myx < (xmin - (dx / 2.0)) || myx > (xmax + (dx / 2.0)) ||
                myy < (ymin - (dy / 2.0)) || myy > (ymax + (dy / 2.0)) ||
                myz < (zmin - (dz / 2.0)) || myz > (zmax + (dz / 2.0))) {
                //if (beam*nrays+raynum == 51) printf("%d %d %d\n", uray <= 0.05 * uray0, myx < (xmin - (dx / 2.0)), myy < (ymin - (dy / 2.0)));
                atomicAdd(counter, 1);
                tt = -1;
                break;                  // "breaks" out of the tt loop once the if condition is satisfied
            }
        }
        if (tt == nt_d) atomicAdd(counter, 1);
        time_passed[beam*nrays+raynum] = (tt == nt_d) ? -1 : tt;
        myx_arr[beam*nrays+raynum] = myx;
        myy_arr[beam*nrays+raynum] = myy;
        myz_arr[beam*nrays+raynum] = myz;
        myvx_arr[beam*nrays+raynum] = myvx;
        myvy_arr[beam*nrays+raynum] = myvy;
        myvz_arr[beam*nrays+raynum] = myvz;
        thisx_0_arr[beam*nrays+raynum] = thisx;
        thisy_0_arr[beam*nrays+raynum] = thisy;
        thisz_0_arr[beam*nrays+raynum] = thisz;
        uray_arr[beam*nrays+raynum] = uray;
    }
}

