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
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, f, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, phi, FFTW_ESTIMATE);

    /* Execute and normalize */
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, N, 0, boxlen);

    /* Apply the inverse Poisson kernel 1/k^2 */
    fft_apply_kernel(fbox, fbox, N, 0, 0, boxlen, kernel_inv_poisson, NULL);

    /* FFT back */
    fft_execute(c2r);
    fft_normalize_c2r(phi, N, N, 0, boxlen);

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
                          const char *out_grid_name, char withELPT,
                          int N, int NX, int X0, long int block_size,
                          double boxlen, MPI_Comm comm) {

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {

        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* Filename of the density grid (should have been stored earlier) */
        char dbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(dbox_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");
        printf("Reading density field '%s'.\n", dbox_fname);

        /* Create 3D arrays */
        double *rho = fftw_alloc_real(NX*N*(N+2));
        fftw_complex *fbox = fftw_alloc_complex(NX*N*(N/2+1));

        /* Load the density field */
        int err = readField_MPI(rho, N, NX, X0, comm, dbox_fname);
        if (err > 0) return err;

        /* Filename of the potential grid that is to be computed */
        char pbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(pbox_fname, "%s/%s_%s%s", pars->OutputDirectory, out_grid_name, Identifier, ".hdf5");

        /* Check if we need to use eLPT or if we can directly solve Poisson's eq */
        if (withELPT && ptype->CyclesOfELPT > 0) {
            /* Base filename for the intermediate step eLPT grids */
            char elptbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(elptbox_fname, "%s/%s_%s", pars->OutputDirectory, ELPT_BASENAME, Identifier);

            /* Solve the Monge-Ampere equation */
            elptChunked(rho, N, boxlen, ptype->CyclesOfELPT, elptbox_fname, pbox_fname);
        } else {
            /* Create MPI FFTW plans */
            fftw_plan r2c_mpi = fftw_mpi_plan_dft_r2c_3d(N, N, N, rho, fbox, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
            fftw_plan c2r_mpi = fftw_mpi_plan_dft_c2r_3d(N, N, N, fbox, rho, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

            /* Execute and normalize */
            fft_execute(r2c_mpi);
            fft_normalize_r2c(fbox, N, NX, X0, boxlen);
            fftw_destroy_plan(r2c_mpi);

            /* Apply the inverse Poisson kernel 1/k^2 */
            fft_apply_kernel(fbox, fbox, N, NX, X0, boxlen, kernel_inv_poisson, NULL);

            /* Execute and normalize */
            fft_execute(c2r_mpi);
            fft_normalize_c2r(rho, N, NX, X0, boxlen);
            fftw_destroy_plan(c2r_mpi);

            /* Export the potential */
            err = writeFieldFile_MPI(rho, N, NX, X0, boxlen, MPI_COMM_WORLD, pbox_fname);
            if (err > 0) return err;
            printf("Potential field written to '%s'.\n", pbox_fname);
        }

        /* Free up memory */
        fftw_free(rho);
        fftw_free(fbox);
    }

    return 0;
}

/* For each particle type, compute derivatives of a certain grid type */
int computeGridDerivatives(const struct params *pars, const struct units *us,
                           const struct cosmology *cosmo,
                           struct particle_type *types, const char *grid_name,
                           const char *out_grid_name, int N, int NX, int X0,
                           long int block_size, double boxlen, MPI_Comm comm) {

    /* Create 3D arrays */
    double *box = fftw_alloc_real(2*block_size);
    fftw_complex *fbox = fftw_alloc_complex(block_size);
    /* Create MPI FFTW plans */
    fftw_plan r2c_mpi = fftw_mpi_plan_dft_r2c_3d(N, N, N, box, fbox, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
    fftw_plan c2r_mpi = fftw_mpi_plan_dft_c2r_3d(N, N, N, fbox, box, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

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
            int err = readField_MPI(box, N, NX, X0, comm, box_fname);
            if (err > 0) return err;

            /* Compute the Fourier transform */
            fft_execute(r2c_mpi);
            fft_normalize_r2c(fbox, N, NX, X0, boxlen);

            /* Compute the derivative */
            fft_apply_kernel(fbox, fbox, N, NX, X0, boxlen, derivatives[i], NULL);

            /* Fourier transform back */
            fft_execute(c2r_mpi);
            fft_normalize_c2r(box, N, NX, X0, boxlen);

            /* Filename of the potential grid */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c_%s%s", pars->OutputDirectory, out_grid_name, letters[i], Identifier, ".hdf5");
            err = writeFieldFile_MPI(box, N, NX, X0, boxlen, MPI_COMM_WORLD, dbox_fname);
            if (err > 0) return err;
            printf("Derivative field written to '%s'.\n", dbox_fname);
        }
    }

    /* Free up memory */
    fftw_free(box);
    fftw_free(fbox);
    fftw_destroy_plan(r2c_mpi);
    fftw_destroy_plan(c2r_mpi);

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
