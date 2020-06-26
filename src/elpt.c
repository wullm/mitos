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
#include <gsl/gsl_linalg.h>


#include "../include/elpt.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/output.h"
#include "../include/poisson.h"
#include "../include/calc_powerspec.h"

/* Compute the determinant of the n X n matrix M */
double determinant(double *M, int n) {
    gsl_matrix_view m = gsl_matrix_view_array (M, n, n);
    gsl_permutation *perm = gsl_permutation_alloc (n);
    int sign;

    /* Compute the LU decomposition */
    gsl_linalg_LU_decomp(&m.matrix, perm, &sign);

    double det = gsl_linalg_LU_det(&m.matrix, sign);

    /* Free memory */
    gsl_permutation_free(perm);

    return det;
}

/* Fast 3x3 determinant */
static inline double det3(double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

/* Solve the Poisson equation D.phi = f using FFT */
int elpt(double *phi, const double *f, int N, double boxlen) {

    /* Create 3D arrays for the source function and its Fourier transform */
    double *box =  calloc(N*N*N, sizeof(double));
    double *box0 =  calloc(N*N*N, sizeof(double));
    double *box2 =  calloc(N*N*N, sizeof(double));
    double *box3 =  calloc(N*N*N, sizeof(double));

    double *box_xx =  calloc(N*N*N, sizeof(double));
    double *box_xy =  calloc(N*N*N, sizeof(double));
    double *box_xz =  calloc(N*N*N, sizeof(double));
    double *box_yy =  calloc(N*N*N, sizeof(double));
    double *box_yz =  calloc(N*N*N, sizeof(double));
    double *box_zz =  calloc(N*N*N, sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_xx = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_xy = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_xz = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_yy = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_yz = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox_zz = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box0, fbox, FFTW_ESTIMATE);

    fftw_plan c2r_xx = fftw_plan_dft_c2r_3d(N, N, N, fbox_xx, box_xx, FFTW_ESTIMATE);
    fftw_plan c2r_xy = fftw_plan_dft_c2r_3d(N, N, N, fbox_xy, box_xy, FFTW_ESTIMATE);
    fftw_plan c2r_xz = fftw_plan_dft_c2r_3d(N, N, N, fbox_xz, box_xz, FFTW_ESTIMATE);
    fftw_plan c2r_yy = fftw_plan_dft_c2r_3d(N, N, N, fbox_yy, box_yy, FFTW_ESTIMATE);
    fftw_plan c2r_yz = fftw_plan_dft_c2r_3d(N, N, N, fbox_yz, box_yz, FFTW_ESTIMATE);
    fftw_plan c2r_zz = fftw_plan_dft_c2r_3d(N, N, N, fbox_zz, box_zz, FFTW_ESTIMATE);

    solvePoisson(box3, f, N, boxlen);

    for (int i=0; i<N*N*N; i++) {
        box[i] = box3[i];
    }


    for (int i=0; i<3; i++) {
        memcpy(box0, box, N*N*N*sizeof(double));

        fft_execute(r2c);
        fft_normalize_r2c(fbox, N, boxlen);

        memcpy(fbox_xx, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        memcpy(fbox_xy, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        memcpy(fbox_xz, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        memcpy(fbox_yy, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        memcpy(fbox_yz, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        memcpy(fbox_zz, fbox, N*N*(N/2+1)*sizeof(fftw_complex));

        fft_apply_kernel(fbox_xx, fbox_xx, N, boxlen, kernel_dx, NULL);
        fft_apply_kernel(fbox_xx, fbox_xx, N, boxlen, kernel_dx, NULL);

        fft_apply_kernel(fbox_xy, fbox_xy, N, boxlen, kernel_dx, NULL);
        fft_apply_kernel(fbox_xy, fbox_xy, N, boxlen, kernel_dy, NULL);

        fft_apply_kernel(fbox_xz, fbox_xz, N, boxlen, kernel_dx, NULL);
        fft_apply_kernel(fbox_xz, fbox_xz, N, boxlen, kernel_dz, NULL);

        fft_apply_kernel(fbox_yy, fbox_yy, N, boxlen, kernel_dy, NULL);
        fft_apply_kernel(fbox_yy, fbox_yy, N, boxlen, kernel_dy, NULL);

        fft_apply_kernel(fbox_yz, fbox_yz, N, boxlen, kernel_dy, NULL);
        fft_apply_kernel(fbox_yz, fbox_yz, N, boxlen, kernel_dz, NULL);

        fft_apply_kernel(fbox_zz, fbox_zz, N, boxlen, kernel_dz, NULL);
        fft_apply_kernel(fbox_zz, fbox_zz, N, boxlen, kernel_dz, NULL);

        fft_execute(c2r_xx);
        fft_normalize_c2r(box_xx, N, boxlen);
        fft_execute(c2r_xy);
        fft_normalize_c2r(box_xy, N, boxlen);
        fft_execute(c2r_xz);
        fft_normalize_c2r(box_xz, N, boxlen);
        fft_execute(c2r_yy);
        fft_normalize_c2r(box_yy, N, boxlen);
        fft_execute(c2r_yz);
        fft_normalize_c2r(box_yz, N, boxlen);
        fft_execute(c2r_zz);
        fft_normalize_c2r(box_zz, N, boxlen);

        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    double d_dxx, d_dyy, d_dzz;
                    double d_dxy, d_dxz, d_dyz;

                    d_dxx = box_xx[row_major(x,y,z,N)];
                    d_dxy = box_xy[row_major(x,y,z,N)];
                    d_dxz = box_xz[row_major(x,y,z,N)];
                    d_dyy = box_yy[row_major(x,y,z,N)];
                    d_dyz = box_yz[row_major(x,y,z,N)];
                    d_dzz = box_zz[row_major(x,y,z,N)];

                    double M[] = {1+d_dxx, d_dxy, d_dxz,
                                  d_dxy, 1+d_dyy, d_dyz,
                                  d_dxz, d_dyz, 1+d_dzz};

                    double det = det3(M);

                    box2[row_major(x,y,z,N)] = (1 + f[row_major(x,y,z,N)]) - det;
                    // phi[row_major(x,y,z,N)] = (det-1 - f[row_major(x,y,z,N)],2)/f[row_major(x,y,z,N)];
                }
            }
        }

        // int bins = 50;
        // double *k_in_bins = malloc(bins * sizeof(double));
        // double *power_in_bins = malloc(bins * sizeof(double));
        // int *obs_in_bins = calloc(bins, sizeof(int));
        //
        // /* Transform back to momentum space */
        // fftw_complex *gogo = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
        // fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, f, gogo, FFTW_ESTIMATE);
        // fft_execute(r2c);
        // fft_normalize_r2c(gogo,N,boxlen);
        // calc_cross_powerspec(N, boxlen, gogo, gogo, bins, k_in_bins, power_in_bins, obs_in_bins);
        //
        // /* Check that it is right */
        // printf("\n");
        // printf("k P_measured(k) P_input(k) observations\n");
        // for (int i=0; i<bins; i++) {
        //     if (obs_in_bins[i] == 0) continue; //skip empty bins
        //
        //     /* The power we observe */
        //     double k = k_in_bins[i];
        //     double Pk = power_in_bins[i];
        //     int obs = obs_in_bins[i];
        //
        //     printf("%f %e %d\n", k, Pk, obs);
        // }
        // printf("\n");


        solvePoisson(box3, box2, N, boxlen);

        for (int i=0; i<N*N*N; i++) {
            box[i] += box3[i];
        }
    }

    // for (int i=0; i<N*N*N; i++) {
    //     phi[i] = box[i];
    // }

    /* Free the memory */
    free(box);
    free(box0);
    free(box2);
    free(box3);

    free(box_xx);
    free(box_xy);
    free(box_xz);
    free(box_yy);
    free(box_yz);
    free(box_zz);

    free(fbox_xx);
    free(fbox_xy);
    free(fbox_xz);
    free(fbox_yy);
    free(fbox_yz);
    free(fbox_zz);

    return 0;
}

/* Solve the Poisson equation D.phi = f using FFT */
int elptChunked(double *phi, const double *f, int N, double boxlen) {

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

    /* Initial guess */
    solvePoisson(box, f, N, boxlen);

    int chunks = 8;
    int chunk_size = N*N*N/chunks;

    writeFieldHeader_H5(N, boxlen, chunks, "rho.hdf5");
    writeField_H5(f, "rho.hdf5");
    writeFieldHeader_H5(N, boxlen, chunks, "resid.hdf5");
    writeFieldHeader_H5(N, boxlen, chunks, "resid_poisson.hdf5");
    writeFieldHeader_H5(N, boxlen, chunks, "phi_step.hdf5");
    writeField_H5(box, "phi_step.hdf5");

    for (int i=0; i<3; i++) {

        // writeGRF_H5(box, N, boxlen, "phi_step.hdf5");

        /* Compute the 6 derivative components of the Hessian */
        for (int j=0; j<6; j++) {
            readGRF_inPlace_H5(box, "phi_step.hdf5");

            fft_execute(r2c);
            fft_normalize_r2c(fbox, N, boxlen);

            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_a[j]], NULL);
            fft_apply_kernel(fbox, fbox, N, boxlen, derivatives[index_b[j]], NULL);

            fft_execute(c2r);
            fft_normalize_c2r(box, N, boxlen);

            char outname[40];
            sprintf(outname, "dphi_d%c%c.hdf5", letters[index_a[j]], letters[index_b[j]]);

            // writeGRF_H5(box, N, boxlen, outname);
            // readGRF_inPlace_H5(box, "phi_step.hdf5");

            writeFieldHeader_H5(N, boxlen, chunks, outname);
            writeField_H5(box, outname);
        }


        // readGRF_inPlace_H5(box, "phi_step.hdf5");

        double *chunk_xx = malloc(chunk_size * sizeof(double));
        double *chunk_xy = malloc(chunk_size * sizeof(double));
        double *chunk_xz = malloc(chunk_size * sizeof(double));
        double *chunk_yy = malloc(chunk_size * sizeof(double));
        double *chunk_yz = malloc(chunk_size * sizeof(double));
        double *chunk_zz = malloc(chunk_size * sizeof(double));
        double *chunk_resid = malloc(chunk_size * sizeof(double));
        double *chunk_rho = malloc(chunk_size * sizeof(double));

        for (int j=0; j<chunks; j++) {
            /* Read the chunks */
            readFieldChunk_H5(chunk_xx, N, chunks, j, "dphi_dxx.hdf5");
            readFieldChunk_H5(chunk_xy, N, chunks, j, "dphi_dxy.hdf5");
            readFieldChunk_H5(chunk_xz, N, chunks, j, "dphi_dxz.hdf5");
            readFieldChunk_H5(chunk_yy, N, chunks, j, "dphi_dyy.hdf5");
            readFieldChunk_H5(chunk_yz, N, chunks, j, "dphi_dyz.hdf5");
            readFieldChunk_H5(chunk_zz, N, chunks, j, "dphi_dzz.hdf5");
            readFieldChunk_H5(chunk_rho, N, chunks, j, "rho.hdf5");

            /* Compute the determinant and store the residual */
            for (int k=0; k<chunk_size; k++) {
                double d_dxx, d_dyy, d_dzz;
                double d_dxy, d_dxz, d_dyz;

                d_dxx = chunk_xx[k];
                d_dxy = chunk_xy[k];
                d_dxz = chunk_xz[k];
                d_dyy = chunk_yy[k];
                d_dyz = chunk_yz[k];
                d_dzz = chunk_zz[k];

                double M[] = {1+d_dxx, d_dxy, d_dxz,
                              d_dxy, 1+d_dyy, d_dyz,
                              d_dxz, d_dyz, 1+d_dzz};

                double det = det3(M);

                chunk_resid[k] = (1 + chunk_rho[k]) - det;
                // chunk_resid[k] = d_dxx;
            }

            /* Store the residual chunk */
            writeFieldChunk_H5(chunk_resid, N, chunks, j, "resid.hdf5");
        }

        free(chunk_xx);
        free(chunk_xy);
        free(chunk_xz);
        free(chunk_yy);
        free(chunk_yz);
        free(chunk_zz);
        free(chunk_resid);


        readGRF_inPlace_H5(box, "resid.hdf5");
        solvePoisson(box, box, N, boxlen);
        writeField_H5(box, "resid_poisson.hdf5");


        double *chunk_add = malloc(chunk_size * sizeof(double));
        double *chunk_phi = malloc(chunk_size * sizeof(double));

        for (int j=0; j<chunks; j++) {
            /* Read the chunks */
            readFieldChunk_H5(chunk_add, N, chunks, j, "resid_poisson.hdf5");
            readFieldChunk_H5(chunk_phi, N, chunks, j, "phi_step.hdf5");

            /* Compute the determinant and store the residual */
            for (int k=0; k<chunk_size; k++) {
                chunk_phi[k] += chunk_add[k];
            }

            /* Store the residual chunk */
            writeFieldChunk_H5(chunk_phi, N, chunks, j, "phi_step.hdf5");
        }

        free(chunk_add);
        free(chunk_phi);


        // for (int x=0; x<N; x++) {
        //     for (int y=0; y<N; y++) {
        //         for (int z=0; z<N; z++) {
        //             double d_dxx, d_dyy, d_dzz;
        //             double d_dxy, d_dxz, d_dyz;
        //
        //             d_dxx = box_xx[row_major(x,y,z,N)];
        //             d_dxy = box_xy[row_major(x,y,z,N)];
        //             d_dxz = box_xz[row_major(x,y,z,N)];
        //             d_dyy = box_yy[row_major(x,y,z,N)];
        //             d_dyz = box_yz[row_major(x,y,z,N)];
        //             d_dzz = box_zz[row_major(x,y,z,N)];
        //
        //             double M[] = {1+d_dxx, d_dxy, d_dxz,
        //                           d_dxy, 1+d_dyy, d_dyz,
        //                           d_dxz, d_dyz, 1+d_dzz};
        //
        //             double det = det3(M);
        //
        //             box2[row_major(x,y,z,N)] = (1 + f[row_major(x,y,z,N)]) - det;
        //             // phi[row_major(x,y,z,N)] = (det-1 - f[row_major(x,y,z,N)],2)/f[row_major(x,y,z,N)];
        //         }
        //     }
        // }

    }

    // for (int i=0; i<N*N*N; i++) {
    //     phi[i] = box[i];
    // }

    /* Free the memory */
    free(box);
    free(fbox);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    return 0;
}
