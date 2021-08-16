#ifndef __DEF_H_
#define __DEF_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <tuple>
#include <assert.h>

#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include <boost/cstdlib.hpp>
#include <boost/next_prior.hpp>
using namespace std;

/* Define flags for compilation */
#define RAY_TRACKER_DIAGNOSTICS 0
#define INTERSECTION_DIAGNOSTICS 0
#define CBET_GAIN_DIAGNOSTICS 0
#define CBET_UPDATE_DIAGNOSTICS 0

/* Define constants */

#define nr 443
/* Deine the 2D cartesian extent of the grid in cm (1e-4 is 1 micron). */
#define xyz_size 100
#define nx xyz_size
#define xmin -0.13
#define xmax 0.13
#define dx ((xmax-xmin)/(nx-1))

#define ny xyz_size
#define ymin -0.13
#define ymax 0.13
#define dy ((ymax-ymin)/(ny-1))

#define nz xyz_size
#define zmin -0.13
#define zmax 0.13
#define dz ((zmax-zmin)/(nz-1))

#define xres ((xmax-xmin)/(nx-1))
#define yres ((ymax-ymin)/(ny-1))
#define zres ((zmax-zmin)/(nz-1))

#define beam_min_x -450.0e-4
#define beam_max_x 450.0e-4

#define nbeams 60

/* Define some constants to be used later */
#define c 29979245800.0 	// speed of light in cm/s
#define e0 8.85418782e-12	// permittivity of free space in m^-3 kg^-1 s^4 A^2
#define me 9.10938356e-31	// electron mass in kg
#define ec 1.60217662e-19	// electron charge in C

#define lambda (1.053e-4/3.0)	// wavelength of light, in cm. This is frequncy-tripled "3w" or "blue" (UV) light
#define freq (c/lambda)		// frequency of light, in Hz
#define omega (2*M_PI*freq)	// frequency of light, in rad/s
#define ncrit (1e-6*(pow(omega, 2)*me*e0/pow(ec, 2)))	// the critical density occurs when omega = omega_p,e

#define rays_per_zone 4 
#define beam_max_x 450.0e-4
#define beam_min_x -450.0e-4
#define nrays_x (int(rays_per_zone*(beam_max_x-beam_min_x)/xres)+1)
#define nrays_y (int(rays_per_zone*(beam_max_x-beam_min_x)/yres)+1)
#define nrays (nrays_x*nrays_y)
#define sigma 0.0375

#define courant_mult 0.5 // 0.37 // 0.25 // 0.36 // 0.22;
#define dt (courant_mult*min(dx,dz)/c)

#if nx >= nz
#define nt (int((1/courant_mult)*nx*2.0))
#else
#define nt (int((1/courant_mult)*nz*2.0))
#endif


#define offset 0.5e-4				//offset = 0.0e-4

#define intensity 1.0e14                     // intensity of the beam in W/cm^2
#define uray_mult (intensity*(courant_mult)*pow(double(pow(rays_per_zone, 2)), -1))

#define numstored (int(5*rays_per_zone))

#define ncrossings (nx*3)	// Maximum number of potential grid crossings by a ray

#define estat 4.80320427e-10 	       // electron charge in statC
#define mach (-1.0*sqrt(2))                 // Mach number for max resonance
#define Z 3.1                        // ionization state
#define mi (10230*(1.0e3*me))          // Mass of ion in g
#define mi_kg (10230.0*me)	   // Mass of ion in kg
#define Te (2.0e3*11604.5052)          // Temperature of electron in K
#define Te_eV 2.0e3
#define Ti (1.0e3*11604.5052)          // Temperature of ion in K
#define Ti_eV 1.0e3
#define iaw 0.2                      // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
#define kb 1.3806485279e-16   //Boltzmann constant in erg/K
#define kb2 1.3806485279e-23   //Boltzmann constant in J/K

#define constant1 ((pow(estat,2))/(4*(1.0e3*me)*c*omega*kb*Te*(1+3*Ti/(Z*Te))))

#define cs (1e2*sqrt(ec*(Z*Te_eV+3.0*Ti_eV)/mi_kg))	// acoustic wave speed, approx. 4e7 cm/s in this example
#define u_flow (machnum*cs)    	// plasma flow velocity

const static int absorption = 1;
#define focal_length 0.1
/* Define Matrix Types*/
typedef boost::multi_array<double, 3> Array3D;
typedef Array3D::index Array3DIdx;
// typedef Array<int,Dynamic,Dynamic> ArrayXXi;
// typedef Array<int,Dynamic> ArrayXi;

// >= 5k
const static long nthreads = min(100000000, nrays*nbeams);
const static int threads_per_block = 256;
const static long threads_per_beam = nthreads/nbeams/threads_per_block;
const static int nindices = ceil(nrays/(float)(threads_per_beam*threads_per_block));

const static int nchunksx = 1;
const static int nchunksy = nchunksx;
const static int nchunksz = nchunksx;
const static int nchunks = nchunksx*nchunksy*nchunksz;
const static int csx = xyz_size/nchunksx;
const static int csy = xyz_size/nchunksy;
const static int csz = xyz_size/nchunksz;
const static int chunk_size = xyz_size/nchunksx;
const static int c3 = csx*csy*csz;

/* Define common operations */
inline unsigned arrSize(tuple<unsigned*, unsigned*> arr)
{
    return unsigned(get<1>(arr) - get<0>(arr));
}

/* Piecewise linear interpolation
   Use binary search to find the segment
   Ref: https://software.llnl.gov/yorick-doc/qref/qrfunc09.html
*/
double interp(const vector<double> y, const vector<double> x, const double xp);

__global__
void chooseRays(int b, int *thisxa, int *thisya, int *thisza, int *timep, int i, int j, int k, int nindices);
__global__
void calculate_myxyz(int nindices, double *beam_norm, double *pow_r,
    double *phase_r, double *uray_arr, double *uray_i, int *time_passed,
    int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
    double *x, double *y, double *z, int *counter,
    double *myx_arr, double *myy_arr, double *myz_arr, double uray_mult_true, double focal_length_true);

__global__
void initial_launch(int nindices, double *wpe, double *beam_norm, double *uray_arr,
    int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
    double *myvx_arr, double *myvy_arr, double *myvz_arr, int *time_passed,
    int i, int j, int k);

__global__
void launch_ray_XYZ(int beam, unsigned nindices,
    double *x, double *y, double *z,
    double *dedendx, double *dedendy, double *dedendz,
    double *edep, double *eden, double *etemp, double *uray_arr,
    double *myx_arr, double *myy_arr, double *myz_arr,
    double *myvx_arr, double *myvy_arr, double *myvz_arr,
    double *uray_init,
    int *thisx_0_arr, int *thisy_0_arr, int *thisz_0_arr,
    int i, int j, int k, int *time_passed, int *update, int *counter);
#endif
