/*******************************************************************************
 * This file is part of DEXM.
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
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/output.h"

/* Solve the Poisson equation D.phi = f using FFT */
int solvePoisson(double *phi, const double *f, int N, double boxlen) {

    /* Create 3D arrays for the source function and its Fourier transform */
    double *box =  (double*) fftw_malloc(N*N*N*sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Copy the source function into the new array, so as to not destroy it */
    memcpy(box, f, N*N*N*sizeof(double));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, phi, FFTW_ESTIMATE);

    /* Execute and normalize */
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Apply the inverse Poisson kernel 1/k^2 */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson);

    /* FFT back */
    fft_execute(c2r);
    fft_normalize_c2r(phi, N, boxlen);

    /* Free all the FFT objects */
    fftw_free(fbox);
    fftw_free(box);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    return 0;
}

/* Solve the Poisson equation for each density grid */
int computePotentialGrids(const struct params *pars, const struct units *us,
                          const struct cosmology *cosmo,
                          struct particle_type *types) {

    /* Grid dimensions */
    const int N = pars->GridSize;
    const double boxlen = pars->BoxLen;

    /* Create 3D arrays */
    double *rho = malloc(N*N*N*sizeof(double));
    double *phi = malloc(N*N*N*sizeof(double));

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the density grid (should have been stored earlier) */
        char dbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(dbox_fname, "%s/%s%s%s", pars->OutputDirectory, "density_", Identifier, ".hdf5");
        printf("Reading density field '%s'.\n", dbox_fname);

        /* Read the density field from file */
        int read_N;
        double read_boxlen;
        readGRF_H5(&rho, &read_N, &read_boxlen, dbox_fname);

        if (N != read_N || boxlen != read_boxlen) {
            printf("Error: grid dimensions do not match file.\n");
            return 1;
        }

        /* Solve Poisson's equation */
        solvePoisson(phi, rho, N, boxlen);

        /* Filename of the potential grid */
        char pbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(pbox_fname, "%s/%s%s%s", pars->OutputDirectory, "potential_", Identifier, ".hdf5");
        printf("Potential field written to '%s'.\n", pbox_fname);
        writeGRF_H5(phi, N, boxlen, pbox_fname);
    }

    /* Free up memory */
    free(rho);
    free(phi);

    return 0;
}

/* Compute derivatives of the potential grids */
int computePotentialDerivatives(const struct params *pars, const struct units *us,
                                const struct cosmology *cosmo,
                                struct particle_type *types) {

    /* Grid dimensions */
    const int N = pars->GridSize;
    const double boxlen = pars->BoxLen;

    /* Arrays and FFT plans */
    double *box =  calloc(N*N*N, sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* We calculate derivatives using FFT kernels */
    const kernel_func derivatives[] = {kernel_dx, kernel_dy, kernel_dz};
    const char letters[] = {'x', 'y', 'z'};

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the density grid (should have been stored earlier) */
        char box_fname[DEFAULT_STRING_LENGTH];
        sprintf(box_fname, "%s/%s%s%s", pars->OutputDirectory, "potential_", Identifier, ".hdf5");

        /* We need all three derivatives */
        for (int i=0; i<3; i++) {
            printf("Reading potential field '%s'.\n", box_fname);

            /* Read the potential field from file */
            readGRF_inPlace_H5(box, box_fname);

            /* Compute the derivative */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[i]);
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Filename of the potential grid */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c_%s%s", pars->OutputDirectory, "displacement", letters[i], Identifier, ".hdf5");
            printf("Displacement field written to '%s'.\n", dbox_fname);
            writeGRF_H5(box, N, boxlen, dbox_fname);
        }
    }

    /* Free up memory */
    free(box);
    free(fbox);
    fftw_destroy_plan(c2r);
    fftw_destroy_plan(r2c);

    return 0;
}
