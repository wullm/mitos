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
#include <gsl/gsl_linalg.h>


#include "../include/elpt.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/output.h"
#include "../include/poisson.h"
#include "../include/calc_powerspec.h"

typedef double* dp;

/* Fast 3x3 determinant */
static inline double det3(double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

/* Solve the Monge-Ampere equation |D.phi| = f using FFT, stopping after a
 * given number of cycles. Store the resulting potential phi as a file
 * at fname. We use chunked grids to facilitate fitting more into the memory. */
int elptChunked(double *f, int N, double boxlen, int cycles, char *basename, char *fname) {

    /* Arrays and FFT plans */
    double *box = calloc(N*N*N, sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* We calculate derivatives using FFT kernels */
    const kernel_func derivatives[] = {kernel_dx, kernel_dy, kernel_dz};
    const char letters[] = {'x', 'y', 'z'};
    /* We need xx, xy, xz, yy, yz, zz to compute the Hessian */
    const int index_a[] = {0, 0, 0, 1, 1, 2};
    const int index_b[] = {0, 1, 2, 1, 2, 2};

    /* Initial guess */
    solvePoisson(box, f, N, boxlen);

    /* Divide the grid into 2^3 chunks to fit more into the memory */
    int chunks = 8;
    int chunk_size = N*N*N/chunks;

    /* Prepare filenames for all the grids used during intermediate steps */
    char rho_fname[DEFAULT_STRING_LENGTH];
    char resid_fname[DEFAULT_STRING_LENGTH];
    char phi_resid_fname[DEFAULT_STRING_LENGTH];
    char dphi_fname[DEFAULT_STRING_LENGTH];
    sprintf(rho_fname, "%s_%s%s", basename, ELPT_RHO, ".hdf5");
    sprintf(resid_fname, "%s_%s%s", basename, ELPT_RESID, ".hdf5");
    sprintf(phi_resid_fname, "%s_%s%s", basename, ELPT_PHI_RESID, ".hdf5");
    sprintf(dphi_fname, "%s_%s", basename, ELPT_DPHI);

    /* Prepare some hdf5 files in the chunked format */
    writeFieldHeader_H5(N, boxlen, chunks, rho_fname);
    writeFieldHeader_H5(N, boxlen, chunks, resid_fname);
    writeFieldHeader_H5(N, boxlen, chunks, phi_resid_fname);
    writeFieldHeader_H5(N, boxlen, chunks, fname);

    /* Store the source grid and the initial best guess unchunked */
    writeField_H5(f, rho_fname);
    writeField_H5(box, fname);

    /* For each eLPT cycle */
    for (int ITER = 0; ITER < cycles; ITER++) {

        /* Compute the 6 derivative components of the Hessian (not chunked) */
        for (int j=0; j<6; j++) {
            /* Read the current best guess of the potential */
            readGRF_inPlace_H5(box, fname);

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d^2 / (dx_i dx_j) */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_a[j]], NULL);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_b[j]], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Store the resulting derivative grid */
            char outname[DEFAULT_STRING_LENGTH];
            sprintf(outname, "%s_d%c%c.hdf5", dphi_fname, letters[index_a[j]], letters[index_b[j]]);
            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }

        /* Allocate memory for one chunk of each of the 6 derivatives */
        double **derivative_chunks = malloc(6 * sizeof(double*));
        for (int j=0; j<6; j++) {
            derivative_chunks[j] = malloc(chunk_size * sizeof(double));
        }

        /* Allocate memory for one chunk of the source grid rho and the residuals */
        double *chunk_rho = malloc(chunk_size * sizeof(double));
        double *chunk_resid = malloc(chunk_size * sizeof(double));

        /* Accumulate the squared residual error and source */
        double eps = 0.d;
        double norm = 0.d;

        /* For each chunk */
        for (int j=0; j<chunks; j++) {

            /* Read a chunk of the 6 derivative components of the Hessian */
            for (int k=0; k<6; k++) {
                /* The filename */
                char inname[DEFAULT_STRING_LENGTH];
                sprintf(inname, "%s_d%c%c.hdf5", dphi_fname, letters[index_a[k]], letters[index_b[k]]);
                readFieldChunk_H5(derivative_chunks[k], N, chunks, j, inname);
            }

            /* Read a chunk of the source grid */
            readFieldChunk_H5(chunk_rho, N, chunks, j, rho_fname);

            /* At each grid point, compute the determinant and store the residual */
            for (int k=0; k<chunk_size; k++) {
                double d_dxx, d_dyy, d_dzz;
                double d_dxy, d_dxz, d_dyz;

                d_dxx = derivative_chunks[0][k];
                d_dxy = derivative_chunks[1][k];
                d_dxz = derivative_chunks[2][k];
                d_dyy = derivative_chunks[3][k];
                d_dyz = derivative_chunks[4][k];
                d_dzz = derivative_chunks[5][k];

                double M[] = {1+d_dxx, d_dxy, d_dxz,
                              d_dxy, 1+d_dyy, d_dyz,
                              d_dxz, d_dyz, 1+d_dzz};

                double det = det3(M);

                chunk_resid[k] = (1 + chunk_rho[k]) - det;

                /* For diagnostics, record the squared residual and source */
                eps += chunk_resid[k] * chunk_resid[k];
                norm += chunk_rho[k] * chunk_rho[k];
            }

            /* Store the residual chunk */
            writeFieldChunk_H5(chunk_resid, N, chunks, j, resid_fname);
        }

        /* Free the memory used for all chunks */
        for (int j=0; j<6; j++) {
            free(derivative_chunks[j]);
        }
        free(derivative_chunks);
        free(chunk_rho);
        free(chunk_resid);

        /* Load the entire residuals grid */
        readGRF_inPlace_H5(box, resid_fname);

        /* Solve the Poisson equation, applied just to the residuals */
        solvePoisson(box, box, N, boxlen);

        /* Store the result */
        writeField_H5(box, phi_resid_fname);

        /* Allocate memory for one chunk of the potential and the residual potential */
        double *chunk_phi = malloc(chunk_size * sizeof(double));
        double *chunk_phi_resid = malloc(chunk_size * sizeof(double));

        /* For each chunk */
        for (int j=0; j<chunks; j++) {
            /* Read a chunk of the potential and the residual potential */
            readFieldChunk_H5(chunk_phi, N, chunks, j, fname);
            readFieldChunk_H5(chunk_phi_resid, N, chunks, j, phi_resid_fname);

            /* Add the residual potential to the potential */
            for (int k=0; k<chunk_size; k++) {
                chunk_phi[k] += chunk_phi_resid[k];
            }

            /* Store the updated potential chunk */
            writeFieldChunk_H5(chunk_phi, N, chunks, j, fname);
        }

        /* Free memory for both chunks */
        free(chunk_phi);
        free(chunk_phi_resid);

        /* Compute the root mean square residual, normalized by the source grid */
        double rms_eps = sqrt((eps / norm) / (N*N*N));
        printf("%03d] Finished eLPT cycle: eps = %e\n", ITER, rms_eps);

    }

    /* Free the memory */
    free(box);
    free(fbox);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    return 0;
}
