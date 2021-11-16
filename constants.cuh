#include "def.cuh"

/*
 * This file is intended to hold constants that are not program parameters
 * but instead constants that are physical constants in some way that 
 * there is never a reason to change
 */

#ifdef DOUBLE
#define nr 443
#define xmin -0.13
#define xmax 0.13
#define dx ((xmax-xmin)/(nx-1))

#define ymin -0.13
#define ymax 0.13
#define dy ((ymax-ymin)/(ny-1))

#define zmin -0.13
#define zmax 0.13
#define dz ((zmax-zmin)/(nz-1))

#define xres ((xmax-xmin)/(nx-1))
#define yres ((ymax-ymin)/(ny-1))
#define zres ((zmax-zmin)/(nz-1))

#define beam_min_x -450.0e-4
#define beam_max_x 450.0e-4

/* Define some constants to be used later */
#define c 29979245800.0 	// speed of light in cm/s
#define e0 8.85418782e-12	// permittivity of free space in m^-3 kg^-1 s^4 A^2
#define me 9.10938356e-31	// electron mass in kg
#define ec 1.60217662e-19	// electron charge in C

#define lambda (1.053e-4/3.0)	// wavelength of light, in cm. This is frequncy-tripled "3w" or "blue" (UV) light
#define freq (c/lambda)		// frequency of light, in Hz
#define omega (2*M_PI*freq)	// frequency of light, in rad/s
#define ncrit (1e-6*(omega*omega)*me*e0/(ec*ec))	// the critical density occurs when omega = omega_p,e

#define rays_per_zone 4 
#define beam_max_x 450.0e-4
#define beam_min_x -450.0e-4

#define nrays_x (int(rays_per_zone*ceil((beam_max_x-beam_min_x)/xres)))
#define nrays_y (int(rays_per_zone*ceil((beam_max_x-beam_min_x)/yres)))
#define nrays (nrays_x*nrays_y)
#define sigma 0.0375

#define courant_mult 0.5 // 0.37 // 0.25 // 0.36 // 0.22;
#define dt (courant_mult*min(dx,dz)/c)
#define nt (int((1/courant_mult)*nz*2.0))

#define offset 0.5e-4				//offset = 0.0e-4

#define intensity 1.0e14                     // intensity of the beam in W/cm^2
#define uray_mult (intensity*(courant_mult)/(double(rays_per_zone*rays_per_zone)))

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

#else

#define nr 443
#define xmin -0.13f
#define xmax 0.13f
#define dx ((xmax-xmin)/(nx-1))

#define ymin -0.13f
#define ymax 0.13f
#define dy ((ymax-ymin)/(ny-1))

#define zmin -0.13f
#define zmax 0.13f
#define dz ((zmax-zmin)/(nz-1))

#define xres ((xmax-xmin)/(nx-1))
#define yres ((ymax-ymin)/(ny-1))
#define zres ((zmax-zmin)/(nz-1))

#define beam_min_x -450.0e-4f
#define beam_max_x 450.0e-4f

#define c 29979245800.0f 	// speed of light in cm/s
#define e0 8.85418782e-12f	// permittivity of free space in m^-3 kg^-1 s^4 A^2
#define me 9.10938356e-31f	// electron mass in kg
#define ec 1.60217662e-19f	// electron charge in C

#define lambda (1.053e-4f/3.0f)	// wavelength of light, in cm. This is frequncy-tripled "3w" or "blue" (UV) light
#define freq (c/lambda)  // frequency of light, in Hz
#define omega (2*M_PI*freq)	// frequency of light, in rad/s
#define ncrit (1e-6f*(omega*omega)*me*e0/(ec*ec))	// the critical density occurs when omega = omega_p,e

#define rays_per_zone 4 

#define nrays_x (int(rays_per_zone*ceil((beam_max_x-beam_min_x)/xres)))
#define nrays_y (int(rays_per_zone*ceil((beam_max_x-beam_min_x)/yres)))
#define nrays (nrays_x*nrays_y)
#define sigma 0.0375f

#define courant_mult 0.5f // 0.37 // 0.25 // 0.36 // 0.22;
#define dt (courant_mult*min(dx,dz)/c)
#define nt (int((1/courant_mult)*nz*2.0f))

#define offset 0.5e-4f  //offset = 0.0e-4

#define intensity 1.0e14f  // intensity of the beam in W/cm^2
#define uray_mult (intensity*(courant_mult)/(float(rays_per_zone*rays_per_zone)))

#define numstored (int(5*rays_per_zone))

#define ncrossings (nx*3)	// Maximum number of potential grid crossings by a ray

#define estat 4.80320427e-10f  // electron charge in statC
#define mach (-1.0*sqrtf(2))  // Mach number for max resonance
#define Z 3.1f  // ionization state
#define mi (10230*(1.0e3f*me))  // Mass of ion in g
#define mi_kg (10230.0f*me)  // Mass of ion in kg
#define Te (2.0e3f*11604.5052f)  // Temperature of electron in K
#define Te_eV 2.0e3f
#define Ti (1.0e3f*11604.5052f)  // Temperature of ion in K
#define Ti_eV 1.0e3f
#define iaw 0.2f  // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
#define kb 1.3806485279e-16f   //Boltzmann constant in erg/K
#define kb2 1.3806485279e-23f   //Boltzmann constant in J/K

#define constant1 ((pow(estat,2f))/(4f*(1.0e3f*me)*c*omega*kb*Te*(1+3*Ti/(Z*Te))))

#define cs (1e2f*sqrt(ec*(Z*Te_eV+3.0f*Ti_eV)/mi_kg))	// acoustic wave speed, approx. 4e7 cm/s in this example
#define u_flow (machnum*cs)    	// plasma flow velocity

#endif

const static long nthreads = min(max_threads, nrays*nbeams);
const static long threads_per_beam = nthreads/nbeams;
const static int nindices = ceil(nrays/(float)(threads_per_beam));
const static long total_size = (long)nx*(long)ny*(long)nz;
const static long edep_size = ((long)nx+2)*((long)ny+2)*((long)nz+2);

