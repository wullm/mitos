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


#include "../include/spt_grid.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/output.h"
#include "../include/poisson.h"
#include "../include/calc_powerspec.h"

/* Solve the Monge-Ampere equation |D.phi| = f using FFT, stopping after a
 * given number of cycles. Store the resulting potential phi as a file
 * at fname. We use chunked grids to facilitate fitting more into the memory. */
int sptChunked(double *f, int N, double boxlen, int cycles, char *basename, char *fname) {

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

    /* For non-symmetric 2-tensors, we need xx, xy, xz, yx=xy, yy, yz, zx=xz, zy=yz, zz */
    const int ns_index_a[] = {0, 0, 0, 0, 1, 1, 0, 1, 2};
    const int ns_index_b[] = {0, 1, 2, 1, 1, 2, 2, 2, 2};

    /* Divide the grid into 2^3 chunks to fit more into the memory */
    int chunks = 8;
    int chunk_size = N*N*N/chunks;

    /* Prepare base filenames for all the intermediate step grids */
    char density_fname[DEFAULT_STRING_LENGTH];
    char gradient_fname[DEFAULT_STRING_LENGTH];
    char flux_fname[DEFAULT_STRING_LENGTH];
    char flux_gradient_fname[DEFAULT_STRING_LENGTH];
    char potential_fname[DEFAULT_STRING_LENGTH];
    char velocity_fname[DEFAULT_STRING_LENGTH];
    char tidal_fname[DEFAULT_STRING_LENGTH];
    char source1_fname[DEFAULT_STRING_LENGTH];
    char source2_fname[DEFAULT_STRING_LENGTH];
    sprintf(density_fname, "%s_%s", basename, SPT_DENSITY);
    sprintf(gradient_fname, "%s_%s", basename, SPT_GRADIENT);
    sprintf(flux_fname, "%s_%s", basename, SPT_FLUX);
    sprintf(flux_gradient_fname, "%s_%s", basename, SPT_FLUX_GRADIENT);
    sprintf(potential_fname, "%s_%s", basename, SPT_POTENTIAL);
    sprintf(velocity_fname, "%s_%s", basename, SPT_VELOCITY);
    sprintf(tidal_fname, "%s_%s", basename, SPT_TIDAL);
    sprintf(source1_fname, "%s_%s_%d%s", basename, SPT_SOURCE, 1, ".hdf5");
    sprintf(source2_fname, "%s_%s_%d%s", basename, SPT_SOURCE, 2, ".hdf5");

    /* Prepare some hdf5 files in the chunked format */
    writeFieldHeader_H5(N, boxlen, chunks, source1_fname);
    writeFieldHeader_H5(N, boxlen, chunks, source2_fname);

    /* Filenames for current density and flux grids */
    char cur_density_fname[DEFAULT_STRING_LENGTH];
    char cur_flux_fname[DEFAULT_STRING_LENGTH];
    sprintf(cur_density_fname, "%s_%03d.hdf5", density_fname, 0);
    sprintf(cur_flux_fname, "%s_%03d.hdf5", flux_fname, 0);

    /* Store the source grid as initial guess for both density and flux */
    writeFieldHeader_H5(N, boxlen, chunks, cur_density_fname);
    writeFieldHeader_H5(N, boxlen, chunks, cur_flux_fname);
    writeField_H5(f, cur_density_fname);
    writeField_H5(f, cur_flux_fname);

    /* For each GridSPT cycle */
    for (int ITER = 0; ITER < cycles; ITER++) {
        /* Current order in perturbation theory */
        int n = ITER + 2;

        /* Update filenames pointing to current density and flux fields */
        sprintf(cur_density_fname, "%s_%03d.hdf5", density_fname, ITER);
        sprintf(cur_flux_fname, "%s_%03d.hdf5", flux_fname, ITER);

        /* Compute the 3 derivatives of the density field */
        for (int j=0; j<3; j++) {
            /* Read the current density field */
            readGRF_inPlace_H5(box, cur_density_fname);

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d / dx_i */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[j], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Store the resulting derivative grid */
            char outname[DEFAULT_STRING_LENGTH];
            sprintf(outname, "%s_d%c_%03d.hdf5", gradient_fname, letters[j], ITER);
            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }

        /* Compute the 3 derivatives of the flux field */
        for (int j=0; j<3; j++) {
            /* Read the current flux field */
            readGRF_inPlace_H5(box, cur_flux_fname);

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d / dx_i */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[j], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Store the resulting derivative grid */
            char outname[DEFAULT_STRING_LENGTH];
            sprintf(outname, "%s_d%c_%03d.hdf5", flux_gradient_fname, letters[j], ITER);
            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }

        /* Solve Poisson's equation, sourced by the current flux field */
        readGRF_inPlace_H5(box, cur_flux_fname);

        /* Solve Poisson's equation */
        solvePoisson(box, box, N, boxlen);

        /* Store the resulting potential grid */
        char cur_potential_fname[DEFAULT_STRING_LENGTH];
        sprintf(cur_potential_fname, "%s_%03d.hdf5", potential_fname, ITER);
        writeFieldHeader_H5(N, boxlen, chunks, cur_potential_fname);
        writeField_H5(box, cur_potential_fname);

        /* Compute the 3 derivatives of the flux potential field */
        for (int j=0; j<3; j++) {
            /* Read the current flux potential field */
            readGRF_inPlace_H5(box, cur_potential_fname);

            /* Fourier transform it */
            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            /* Compute the derivative d / dx_i */
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[j], NULL);

            /* Fourier transform back */
            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            /* Store the resulting derivative grid */
            char outname[DEFAULT_STRING_LENGTH];
            sprintf(outname, "%s_%c_%03d.hdf5", velocity_fname, letters[j], ITER);
            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }

        /* Compute the 6 seond order derivatives of the flux potential field */
        for (int j=0; j<6; j++) {
            /* Read the current flux potential field */
            readGRF_inPlace_H5(box, cur_potential_fname);

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
            sprintf(outname, "%s_d%c%c_%03d.hdf5", tidal_fname, letters[index_a[j]], letters[index_b[j]], ITER);
            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }

        /* Allocate memory for one chunk of the output grid and two input grids */
        double *chunk_output = malloc(chunk_size * sizeof(double));
        double *chunk_input1 = malloc(chunk_size * sizeof(double));
        double *chunk_input2 = malloc(chunk_size * sizeof(double));

        /* For each chunk */
        for (int j=0; j<chunks; j++) {

            /* Reset the output chunk */
            memset(chunk_output, 0, chunk_size * sizeof(double));

            /* For each order m < n */
            for (int m = 1; m < n; m++) {
                /* We combine a grid of order m with a grid of order l = n - m */
                int ITER_m = m - 1;
                int ITER_l = (n - m) - 1;

                printf("Doing %d and %d\n", ITER_m, ITER_l);

                /* Compute the inner product of the gradient and velocity fields */
                for (int d=0; d<3; d++) {
                    /* Read the gradient component */
                    char gradient_in[DEFAULT_STRING_LENGTH];
                    sprintf(gradient_in, "%s_d%c_%03d.hdf5", gradient_fname, letters[d], ITER_m);
                    readFieldChunk_H5(chunk_input1, N, chunks, j, gradient_in);

                    /* Read the velocity component */
                    char velocity_in[DEFAULT_STRING_LENGTH];
                    sprintf(velocity_in, "%s_%c_%03d.hdf5", velocity_fname, letters[d], ITER_l);
                    readFieldChunk_H5(chunk_input2, N, chunks, j, velocity_in);

                    /* At each grid point, compute the dot product */
                    for (int k=0; k<chunk_size; k++) {
                        chunk_output[k] += chunk_input1[k] * chunk_input2[k];
                    }
                }

                /* Read the density field and flux field */
                char density_in[DEFAULT_STRING_LENGTH];
                char flux_in[DEFAULT_STRING_LENGTH];
                sprintf(density_in, "%s_%03d.hdf5", density_fname, ITER_m);
                sprintf(flux_in, "%s_%03d.hdf5", flux_fname, ITER_l);
                readFieldChunk_H5(chunk_input1, N, chunks, j, density_in);
                readFieldChunk_H5(chunk_input2, N, chunks, j, flux_in);

                /* Add the product of the density and flux grids */
                for (int k=0; k<chunk_size; k++) {
                    chunk_output[k] += chunk_input1[k] * chunk_input2[k];
                }
            }

            /* Store the chunk of the first source grid */
            writeFieldChunk_H5(chunk_output, N, chunks, j, source1_fname);

            /* Reset the output chunk */
            memset(chunk_output, 0, chunk_size * sizeof(double));

            /* For each order m < n */
            for (int m = 1; m < n; m++) {
                /* We combine a grid of order m with a grid of order l = n - m */
                int ITER_m = m - 1;
                int ITER_l = (n - m) - 1;

                printf("Doing %d and %d\n", ITER_m, ITER_l);

                /* Compute the inner product of the flux gradient and velocity fields */
                for (int d=0; d<3; d++) {
                    /* Read the flux gradient component */
                    char gradient_in[DEFAULT_STRING_LENGTH];
                    sprintf(gradient_in, "%s_d%c_%03d.hdf5", flux_gradient_fname, letters[d], ITER_m);
                    readFieldChunk_H5(chunk_input1, N, chunks, j, gradient_in);

                    /* Read the velocity component */
                    char velocity_in[DEFAULT_STRING_LENGTH];
                    sprintf(velocity_in, "%s_%c_%03d.hdf5", velocity_fname, letters[d], ITER_l);
                    readFieldChunk_H5(chunk_input2, N, chunks, j, velocity_in);

                    /* At each grid point, compute the dot product */
                    for (int k=0; k<chunk_size; k++) {
                        chunk_output[k] += chunk_input1[k] * chunk_input2[k];
                    }
                }

                /* Compute the inner product of the tidal tensor fields */
                for (int d=0; d<9; d++) {
                    /* Read the first tidal tensor */
                    char tensor_one[DEFAULT_STRING_LENGTH];
                    sprintf(tensor_one, "%s_d%c%c_%03d.hdf5", tidal_fname, letters[ns_index_a[d]], letters[ns_index_b[d]], ITER_m);
                    readFieldChunk_H5(chunk_input1, N, chunks, j, tensor_one);

                    /* Read the second tidal tensor */
                    char tensor_two[DEFAULT_STRING_LENGTH];
                    sprintf(tensor_two, "%s_d%c%c_%03d.hdf5", tidal_fname, letters[ns_index_a[d]], letters[ns_index_b[d]], ITER_l);
                    readFieldChunk_H5(chunk_input2, N, chunks, j, tensor_two);

                    /* At each grid point, compute the product */
                    for (int k=0; k<chunk_size; k++) {
                        chunk_output[k] += chunk_input1[k] * chunk_input2[k];
                    }
                }
            }

            /* Store the chunk of the second source grid */
            writeFieldChunk_H5(chunk_output, N, chunks, j, source2_fname);
        }

        /* Load the source fields and apply low-pass filters */
        readGRF_inPlace_H5(box, source1_fname);

        /* Fourier transform it */
        fft_execute(r2c);
        fft_normalize_r2c(fbox, N, boxlen);

        /* Apply the filter */
        fft_apply_kernel(fbox, fbox, N, boxlen, lowpass, NULL);
        fft_apply_kernel(fbox, fbox, N, boxlen, hipass, NULL);

        /* Fourier transform back */
        fft_execute(c2r);
        fft_normalize_c2r(box, N, boxlen);

        /* Store the resulting derivative grid */
        writeField_H5(box, source1_fname);


        /* Load the source fields and apply low-pass filters */
        readGRF_inPlace_H5(box, source2_fname);

        /* Fourier transform it */
        fft_execute(r2c);
        fft_normalize_r2c(fbox, N, boxlen);

        /* Apply the filter */
        fft_apply_kernel(fbox, fbox, N, boxlen, lowpass, NULL);
        fft_apply_kernel(fbox, fbox, N, boxlen, hipass, NULL);

        /* Fourier transform back */
        fft_execute(c2r);
        fft_normalize_c2r(box, N, boxlen);

        /* Store the resulting derivative grid */
        writeField_H5(box, source2_fname);

        /* Filenames pointing to the updated density and flux fields */
        char next_density_fname[DEFAULT_STRING_LENGTH];
        char next_flux_fname[DEFAULT_STRING_LENGTH];
        sprintf(next_density_fname, "%s_%03d.hdf5", density_fname, ITER + 1);
        sprintf(next_flux_fname, "%s_%03d.hdf5", flux_fname, ITER + 1);

        /* Prepare the new files */
        writeFieldHeader_H5(N, boxlen, chunks, next_density_fname);
        writeFieldHeader_H5(N, boxlen, chunks, next_flux_fname);

        double global_factor = 2.0 / ((2*n + 3) * (n - 1));
        double eps_d = 0.d;
        double eps_t = 0.d;

        /* For each chunk */
        for (int j=0; j<chunks; j++) {
            /* Read the density field into the output grid */
            readFieldChunk_H5(chunk_output, N, chunks, j, cur_density_fname);

            /* Read the source grids into the inout grids */
            readFieldChunk_H5(chunk_input1, N, chunks, j, source1_fname);
            readFieldChunk_H5(chunk_input2, N, chunks, j, source2_fname);

            double factor1 = (n + 0.5) * global_factor;
            double factor2 = 1.0 * global_factor;

            /* Add the sources to the output */
            for (int k=0; k<chunk_size; k++) {
                chunk_output[k] += factor1 * chunk_input1[k];
                chunk_output[k] += factor2 * chunk_input2[k];

                /* For diagnostics */
                eps_d += factor1 * factor1 * chunk_input1[k] *chunk_input1[k];
                eps_d += factor2 * factor2 * chunk_input2[k] *chunk_input2[k];
            }

            /* Store the updated density grid */
            writeFieldChunk_H5(chunk_output, N, chunks, j, next_density_fname);
        }

        /* For each chunk */
        for (int j=0; j<chunks; j++) {
            /* Read the flux field into the output grid */
            readFieldChunk_H5(chunk_output, N, chunks, j, cur_flux_fname);

            /* Read the source grids into the inout grids */
            readFieldChunk_H5(chunk_input1, N, chunks, j, source1_fname);
            readFieldChunk_H5(chunk_input2, N, chunks, j, source2_fname);

            double factor1 = 1.5 * global_factor;
            double factor2 = n * global_factor;

            /* Add the sources to the output */
            for (int k=0; k<chunk_size; k++) {
                chunk_output[k] += factor1 * chunk_input1[k];
                chunk_output[k] += factor2 * chunk_input2[k];

                /* For diagnostics */
                eps_t += factor1 * factor1 * chunk_input1[k] *chunk_input1[k];
                eps_t += factor2 * factor2 * chunk_input2[k] *chunk_input2[k];
            }

            /* Store the updated density grid */
            writeFieldChunk_H5(chunk_output, N, chunks, j, next_flux_fname);
        }

        eps_d = sqrt(eps_d / (N * N * N));
        eps_t = sqrt(eps_t / (N * N * N));

        printf("%03d] Finished SPT cycle, eps = (%e, %e)\n", ITER, eps_d, eps_t);

    }

    /* Free the memory */
    free(box);
    free(fbox);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    return 0;
}
