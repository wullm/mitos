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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <fftw3.h>
#include <sys/time.h>

#include "../../include/mitos.h"

#define outname(s,x) sprintf(s, "%s/%s", pars.OutputDirectory, x);
#define printheader(s) printf("\n%s%s%s\n", TXT_BLUE, s, TXT_RESET);

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printheader("Mitos Initial Condition Generator");
    printf("The parameter file is '%s'\n", fname);

    /* Timer */
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    /* Mitos structuress */
    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);


    /* First, compute the empirical power spectrum <(1+delta_h)v_h, v_m> */
    
    /* We will read the input into these arrays */
    double *deltah_vh_x = NULL; //halo x-momentum
    double *deltah_vh_y = NULL; //halo y-momentum
    double *deltah_vh_z = NULL; //halo z-momentum   
    
    /* Dimensions of the grid */
    int N;
    double boxlen;
    
    /* Read the halo momentum grids */
    char letters[3] = {'x', 'y', 'z'};
    double *grids_h[3] = {deltah_vh_x, deltah_vh_y, deltah_vh_z};
    for (int i=0; i<3; i++) {
        /* Filename of momentum input grid */
        char read_fname[50];
        sprintf(read_fname, "momentum_halos_%c.hdf5", letters[i]);
        printf("Reading input array '%s'.\n", read_fname);
        
        /* Read the grid */
        readFieldFile(&grids_h[i], &N, &boxlen, read_fname);
    }
    
    /* We will read the input into these arrays */
    double *vm_x = NULL; //matter x-velocity
    double *vm_y = NULL; //matter y-velocity
    double *vm_z = NULL; //matter z-velocity
    double *delta_h = NULL; //halo overdensity
    
    /* Read the matter velocity grids */
    double *grids_m[3] = {vm_x, vm_y, vm_z};
    for (int i=0; i<3; i++) {
        /* Filename of velocity input grid */
        char read_fname[50];
        sprintf(read_fname, "velocity_cdm_%c.hdf5", letters[i]);
        printf("Reading input array '%s'.\n", read_fname);
        
        /* Read the grid */
        readFieldFile(&grids_m[i], &N, &boxlen, read_fname);
    }
    
    /* Read the halo overdensity grid */
    char read_fname[50] = "density_halos.hdf5";
    printf("Reading input array '%s'.\n", read_fname);
    
    /* Read the grid */
    readFieldFile(&delta_h, &N, &boxlen, read_fname);
    
    
    /* Compute the empirical power spectrum along each dimension */
    for (int dim = 0; dim < 3; dim++) {
        /* Allocate 3D real arrays */
        double *ph_i = (double*) fftw_malloc(N*N*N*sizeof(double));
        double *vm_i = (double*) fftw_malloc(N*N*N*sizeof(double));
        
        /* Copy the correct data */
        memcpy(ph_i, grids_h[dim], N*N*N*sizeof(double));
        memcpy(vm_i, grids_m[dim], N*N*N*sizeof(double));
        
        /* Allocate 3D complex arrays */
        fftw_complex *f_ph_i = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_complex *f_vm_i = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
        
        /* Create FFT plans */
        fftw_plan r2c_h = fftw_plan_dft_r2c_3d(N, N, N, ph_i, f_ph_i, FFTW_ESTIMATE);
        fftw_plan r2c_m = fftw_plan_dft_r2c_3d(N, N, N, vm_i, f_vm_i, FFTW_ESTIMATE);
        
        /* Execute FFTs and normalize */
        fft_execute(r2c_h);
        fft_execute(r2c_m);
        fft_normalize_r2c(f_ph_i, N, boxlen);
        fft_normalize_r2c(f_vm_i, N, boxlen);
        fftw_destroy_plan(r2c_h);
        fftw_destroy_plan(r2c_m);
        
        /* Allocate power spectrum arrays */
        int bins = pars.PowerSpectrumBins;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));
        
        /* Calculate the power spectrum */
        calc_cross_powerspec(N, boxlen, f_ph_i, f_vm_i, bins, k_in_bins, power_in_bins, obs_in_bins);
        
        printf("\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%e %e %d\n", k, Pk, obs);
        }
        printf("\n");
        
        /* Clean up the grids */
        free(f_ph_i);
        free(f_vm_i);
        free(ph_i);
        free(vm_i);
    }
    
    // /* Compute the S_alpha power spectrum in each bin */
    // for (int bin = 0; bin < 1; bin++) {
    //     /* Specification of the bin */
    //     double k_min = 1.0e-2;
    //     double k_max = 2.0e-2;
    //     double params[2] = {k_min, k_max};
    // 
    //     for (int dim = 0; dim < 3; dim++) {
    //         /* Select the correct array */
    //         double *vm_i = grids_m[dim];
    // 
    //         /* Allocate 3D complex arrays */
    //         fftw_complex *f_ph_i = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    //         fftw_complex *f_vm_i = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    // 
    // 
    //     /* Execute FFTs and normalize */
    //     fft_execute(r2c_x);
    //     fft_execute(r2c_y);
    //     fft_execute(r2c_z);
    //     fft_normalize_r2c(fbox_x, N, boxlen);
    //     fft_normalize_r2c(fbox_y, N, boxlen);
    //     fft_normalize_r2c(fbox_z, N, boxlen);
    // 
    //     /* Apply sharp k-space filter */
    //     fft_apply_kernel(fbox_x, fbox_x, N, boxlen, kernel_tophat, params);
    //     fft_apply_kernel(fbox_y, fbox_y, N, boxlen, kernel_tophat, params);
    //     fft_apply_kernel(fbox_z, fbox_z, N, boxlen, kernel_tophat, params);
    // 
    //     /* Execute reverse FFTs and normalize */
    //     fft_execute(c2r_x);
    //     fft_execute(c2r_y);
    //     fft_execute(c2r_z);
    //     fft_normalize_c2r(v_alpha_x, N, boxlen);
    //     fft_normalize_c2r(v_alpha_y, N, boxlen);
    //     fft_normalize_c2r(v_alpha_z, N, boxlen);
    // 
    //     /* In real space, multiply by the halo overdensity grid */
    //     for (int i=0; i<N*N*N; i++) {
    //         v_alpha_x[i] *= delta_h[i];
    //     }
    // 
    //     /* Execute FFTs and normalize */
    //     fft_execute(r2c_x_2);
    //     fft_execute(r2c_y_2);
    //     fft_execute(r2c_z_2);
    //     fft_normalize_r2c(fbox_x, N, boxlen);
    //     fft_normalize_r2c(fbox_y, N, boxlen);
    //     fft_normalize_r2c(fbox_z, N, boxlen);
    // 
    //     /* Compute power spectra and sum */
    // 
    // }
    
    // /* Allocate 3D complex arrays */
    // fftw_complex *fbox_x = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    // fftw_complex *fbox_y = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    // fftw_complex *fbox_z = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    // 
    // /* Allocate more 3D arrays */
    // double *v_alpha_x = (double*) fftw_malloc(N*N*N*sizeof(double));
    // double *v_alpha_y = (double*) fftw_malloc(N*N*N*sizeof(double));
    // double *v_alpha_z = (double*) fftw_malloc(N*N*N*sizeof(double));
    // 
    // /* Create FFT plans */
    // fftw_plan r2c_x = fftw_plan_dft_r2c_3d(N, N, N, v_m_x, fbox_x, FFTW_ESTIMATE);
    // fftw_plan r2c_y = fftw_plan_dft_r2c_3d(N, N, N, v_m_y, fbox_y, FFTW_ESTIMATE);
    // fftw_plan r2c_z = fftw_plan_dft_r2c_3d(N, N, N, v_m_z, fbox_z, FFTW_ESTIMATE);
    // fftw_plan c2r_x = fftw_plan_dft_c2r_3d(N, N, N, fbox_x, v_alpha_x, FFTW_ESTIMATE);
    // fftw_plan c2r_y = fftw_plan_dft_c2r_3d(N, N, N, fbox_y, v_alpha_y, FFTW_ESTIMATE);
    // fftw_plan c2r_z = fftw_plan_dft_c2r_3d(N, N, N, fbox_z, v_alpha_z, FFTW_ESTIMATE);
    // fftw_plan r2c_x_2 = fftw_plan_dft_r2c_3d(N, N, N, v_alpha_x, fbox_x, FFTW_ESTIMATE);
    // fftw_plan r2c_y_2 = fftw_plan_dft_r2c_3d(N, N, N, v_alpha_y, fbox_y, FFTW_ESTIMATE);
    // fftw_plan r2c_z_2 = fftw_plan_dft_r2c_3d(N, N, N, v_alpha_z, fbox_z, FFTW_ESTIMATE);
    // 
    // /* Compute the S_alpha power spectrum in each bin */
    // for (int bin = 0; bin < 8; bin++) {
    //     /* Specification of the bin */
    //     double k_min = 1.0e-2;
    //     double k_max = 2.0e-2;
    //     double params[2] = {k_min, k_max};
    // 
    //     /* Execute FFTs and normalize */
    //     fft_execute(r2c_x);
    //     fft_execute(r2c_y);
    //     fft_execute(r2c_z);
    //     fft_normalize_r2c(fbox_x, N, boxlen);
    //     fft_normalize_r2c(fbox_y, N, boxlen);
    //     fft_normalize_r2c(fbox_z, N, boxlen);
    // 
    //     /* Apply sharp k-space filter */
    //     fft_apply_kernel(fbox_x, fbox_x, N, boxlen, kernel_tophat, params);
    //     fft_apply_kernel(fbox_y, fbox_y, N, boxlen, kernel_tophat, params);
    //     fft_apply_kernel(fbox_z, fbox_z, N, boxlen, kernel_tophat, params);
    // 
    //     /* Execute reverse FFTs and normalize */
    //     fft_execute(c2r_x);
    //     fft_execute(c2r_y);
    //     fft_execute(c2r_z);
    //     fft_normalize_c2r(v_alpha_x, N, boxlen);
    //     fft_normalize_c2r(v_alpha_y, N, boxlen);
    //     fft_normalize_c2r(v_alpha_z, N, boxlen);
    // 
    //     /* In real space, multiply by the halo overdensity grid */
    //     for (int i=0; i<N*N*N; i++) {
    //         v_alpha_x[i] *= delta_h[i];
    //     }
    // 
    //     /* Execute FFTs and normalize */
    //     fft_execute(r2c_x_2);
    //     fft_execute(r2c_y_2);
    //     fft_execute(r2c_z_2);
    //     fft_normalize_r2c(fbox_x, N, boxlen);
    //     fft_normalize_r2c(fbox_y, N, boxlen);
    //     fft_normalize_r2c(fbox_z, N, boxlen);
    // 
    //     /* Compute power spectra and sum */
    // 
    // }
    
    
    // 
    // /* Create FFT plans */
    // fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, rho, fbox, FFTW_ESTIMATE);
    // fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, rho, FFTW_ESTIMATE);
    // 
    // /* Execute and normalize */
    // fft_execute(r2c);
    // fft_normalize_r2c(fbox, N, boxlen);
    // 
    // /* What transfer function should we apply? */
    // int index_src = findTitle(ptdat.titles, "d_ncdm[0]", ptdat.n_functions);
    // 
    // /* Package the spline parameters */
    // struct spline_params sp = {&spline, index_src, tau_index, u_tau};
    // 
    // /* Apply the transfer function to fbox */
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_transfer_function, &sp);
    // 
    // sp.index_src = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
    // fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_transfer_function, &sp);
    // 
    // /* Execute and normalize */
    // fft_execute(c2r);
    // fft_normalize_c2r(rho, N, boxlen);
    // 
    // /* Export the real box */
    // writeFieldFile(rho, N, boxlen, pars.OutputFilename);
    // printf("Resulting field exported to '%s'.\n", pars.OutputFilename);

    // 
    // /* Free up memory */
    // free(rho);
    // fftw_destroy_plan(r2c);

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);

    /* Timer */
    gettimeofday(&stop, NULL);
    long unsigned microsec = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\nTime elapsed: %.3f ms\n", microsec/1000.);

}
