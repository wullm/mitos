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
#include "../include/dexm.h"

/* Solve the Poisson equation D.phi = f using FFT */
int solvePoisson(double *phi, double *f, int N, double boxlen) {

    /* Create 3D arrays for the source function and its Fourier transform */
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

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

/* Solve the Poisson equation for each source grid */
int computePotentialGrids(const struct params *pars, const struct units *us,
                          const struct cosmology *cosmo,
                          struct particle_type *types, const char *grid_name,
                          const char *out_grid_name, char withELPT) {

    /* Grid dimensions */
    const int N = pars->GridSize;
    const double boxlen = pars->BoxLen;

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
        /* We will read the density into this array and do in-place calculations */
        double *rho;

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the density grid (should have been stored earlier) */
        char dbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(dbox_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");
        printf("Reading density field '%s'.\n", dbox_fname);

        /* Read the density field from file */
        int read_N;
        double read_boxlen;
        readGRF_H5(&rho, &read_N, &read_boxlen, dbox_fname);

        if (N != read_N || boxlen != read_boxlen) {
            printf("Error: grid dimensions do not match file.\n");
            return 1;
        }

        /* Filename of the potential grid that is to be computed */
        char pbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(pbox_fname, "%s/%s_%s%s", pars->OutputDirectory, out_grid_name, Identifier, ".hdf5");

        /* Check if we need to use eLPT or if we can directly solve Poisson's eq */
        if (withELPT && ptype->CyclesOfELPT > 0) {
            /* Base filename for the intermediate step eLPT grids */
            char elptbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(elptbox_fname, "%s/%s_%s", pars->OutputDirectory, "elpt", Identifier);

            /* Solve the Monge-Ampere equation */
            elptChunked(rho, N, boxlen, ptype->CyclesOfELPT, elptbox_fname, pbox_fname);
        } else {
            /* Solve Poisson's equation and store the result back in rho */
            solvePoisson(rho, rho, N, boxlen);

            /* Export the potential */
            printf("Potential field written to '%s'.\n", pbox_fname);
            writeGRF_H5(rho, N, boxlen, pbox_fname);
        }

        /* Free up memory */
        free(rho);
    }

    return 0;
}

/* For each particle type, compute derivatives of a certain grid type */
int computeGridDerivatives(const struct params *pars, const struct units *us,
                           const struct cosmology *cosmo,
                           struct particle_type *types, const char *grid_name,
                           const char *out_grid_name) {

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

    /* For each particle type, read the desired field and compute derivatives */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the desired grid (should have been stored earlier) */
        char box_fname[DEFAULT_STRING_LENGTH];
        sprintf(box_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");

        /* We need all three derivatives */
        for (int i=0; i<3; i++) {
            printf("Reading input field '%s'.\n", box_fname);

            /* Read the potential field from file */
            int err = readGRF_inPlace_H5(box, box_fname);
            if (err > 0) return err;

            /* Compute the Fourier transform */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[i], NULL);

            /* Undo the TSC window function */
            // undoNGPWindow(fbox, N, boxlen);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Filename of the potential grid */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c_%s%s", pars->OutputDirectory, out_grid_name, letters[i], Identifier, ".hdf5");
            printf("Derivative field written to '%s'.\n", dbox_fname);
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



/* For each particle type, compute derivatives of a certain grid type */
int computeSecondGridDerivatives(const struct params *pars, const struct units *us,
                                 const struct cosmology *cosmo,
                                 struct particle_type *types, const char *grid_name,
                                 const char *out_grid_name) {

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
    /* We need xx, xy, xz, yy, yz, zz to compute the Hessian */
    const int index_a[] = {0, 0, 0, 1, 1, 2};
    const int index_b[] = {0, 1, 2, 1, 2, 2};

    /* For each particle type, read the desired field and compute derivatives */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the desired grid (should have been stored earlier) */
        char box_fname[DEFAULT_STRING_LENGTH];
        sprintf(box_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");

        /* Compute the 6 derivative components of the Hessian (not chunked) */
        for (int i=0; i<6; i++) {
            printf("Reading input field '%s'.\n", box_fname);

            /* Read the potential field from file */
            int err = readGRF_inPlace_H5(box, box_fname);
            if (err > 0) return err;

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d^2 / (dx_i dx_j) */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_a[i]], NULL);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_b[i]], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Filename of the potential grid */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c%c_%s%s", pars->OutputDirectory, out_grid_name, letters[index_a[i]], letters[index_b[i]], Identifier, ".hdf5");
            printf("Derivative field written to '%s'.\n", dbox_fname);
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

/* For each particle type, compute derivatives of a certain grid type */
int computeThirdGridDerivatives(const struct params *pars, const struct units *us,
                                const struct cosmology *cosmo,
                                struct particle_type *types, const char *grid_name,
                                const char *out_grid_name) {

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
    /* We need xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz */
    const int index_a[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2};
    const int index_b[] = {0, 0, 0, 1, 1, 2, 1, 1, 2, 2};
    const int index_c[] = {0, 1, 2, 1, 2, 2, 1, 2, 2, 2};

    /* For each particle type, read the desired field and compute derivatives */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the desired grid (should have been stored earlier) */
        char box_fname[DEFAULT_STRING_LENGTH];
        sprintf(box_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");

        /* Compute the 10 derivatives */
        for (int i=0; i<10; i++) {
            printf("Reading input field '%s'.\n", box_fname);

            /* Read the potential field from file */
            int err = readGRF_inPlace_H5(box, box_fname);
            if (err > 0) return err;

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d^2 / (dx_i dx_j) */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_a[i]], NULL);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_b[i]], NULL);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_c[i]], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Filename of the potential grid */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c%c%c_%s%s", pars->OutputDirectory, out_grid_name, letters[index_a[i]], letters[index_b[i]], letters[index_c[i]], Identifier, ".hdf5");
            printf("Derivative field written to '%s'.\n", dbox_fname);
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
