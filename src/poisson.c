/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/poisson.h"
#include "../include/mitos.h"

/* Solve the Poisson equation D.phi = f using FFT */
int solvePoisson(double *phi, double *f, int N, double boxlen) {

    /* Create 3D arrays for the source function and its Fourier transform */
    fftw_complex *fbox = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, f, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, phi, FFTW_ESTIMATE);

    /* Execute and normalize */
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Apply the inverse Poisson kernel 1/k^2 */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson, NULL);

    /* FFT back */
    fft_execute(c2r);
    fft_normalize_c2r(phi, N, boxlen);

    /* Free all the FFT objects */
    fftw_free(fbox);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    return 0;
}

/* Solve the Poisson equation D.phi = f using FFT */
int solvePoisson_dg(struct distributed_grid *dg) {

    /* Perform Fourier transform on the real array */
    fft_r2c_dg(dg);

    /* Apply the inverse Poisson kernel 1/k^2 */
    fft_apply_kernel_dg(dg, dg, kernel_inv_poisson, NULL);

    /* FFT back */
    fft_c2r_dg(dg);

    return 0;
}


/* Compute higher order perturbed grids for each particle type */
int computePerturbedGrids(const struct params *pars, const struct units *us,
                          const struct cosmology *cosmo,
                          struct particle_type *types,
                          const char *density_grid_name,
                          const char *flux_density_grid_name) {

    /* Grid dimensions */
    const int N = pars->GridSize;
    const double boxlen = pars->BoxLen;


    /* For each particle type */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Skip if we don't need SPT at all */
        if (ptype->CyclesOfSPT == 0) continue;

        /* Filename of the density grid (should have been stored earlier) */
        char density_fname[DEFAULT_STRING_LENGTH];
        sprintf(density_fname, "%s/%s_%s%s", pars->OutputDirectory, density_grid_name, Identifier, ".hdf5");

        /* Filename of the flux density grid (should have been stored earlier) */
        char flux_density_fname[DEFAULT_STRING_LENGTH];
        sprintf(flux_density_fname, "%s/%s_%s%s", pars->OutputDirectory, flux_density_grid_name, Identifier, ".hdf5");

        /* Base filename for the intermediate step SPT grids */
        char sptbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(sptbox_fname, "%s/%s_%s", pars->OutputDirectory, SPT_BASENAME, Identifier);

        /* Execute the SPT program for the desired number of cycles */
        sptChunked(N, boxlen, ptype->CyclesOfSPT, sptbox_fname, density_fname, flux_density_fname, density_fname, flux_density_fname);

    }

    return 0;
}
