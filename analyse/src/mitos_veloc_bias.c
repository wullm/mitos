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
#include <float.h>
#include <fftw3.h>
#include <sys/time.h>
#include <gsl/gsl_linalg.h>

#include "../../include/mitos.h"

#define outname(s,x) sprintf(s, "%s/%s", pars.OutputDirectory, x);
#define printheader(s) printf("\n%s%s%s\n", TXT_BLUE, s, TXT_RESET);

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }
        
    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

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


    /* Open the Halos file */
    message(rank, "Reading halos from '%s'.\n", pars.HaloInputFilename);
    hid_t h_halo_file = openFile_MPI(MPI_COMM_WORLD, pars.HaloInputFilename);

    /* Open the halo masses dataset */
    hid_t h_halo_dat = H5Dopen2(h_halo_file, "Mvir", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_halo_space = H5Dget_space(h_halo_dat);
    hsize_t halo_num;
    H5Sget_simple_extent_dims(h_halo_space, &halo_num, NULL);
    H5Sclose(h_halo_space);

    /* Allocate memory for the halo masses and coordinates */
    double *halo_M = malloc(halo_num * sizeof(double));
    double *halo_x = malloc(halo_num * sizeof(double));
    double *halo_y = malloc(halo_num * sizeof(double));
    double *halo_z = malloc(halo_num * sizeof(double));
    double *halo_vx = malloc(halo_num * sizeof(double));
    double *halo_vy = malloc(halo_num * sizeof(double));
    double *halo_vz = malloc(halo_num * sizeof(double));
    long long *hostHaloId = malloc(halo_num * sizeof(long long));

    message(rank, "We have %lld halos\n", halo_num);

    /* Read out the masses */
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_M);
    H5Dclose(h_halo_dat);

    /* Read out the X-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Xcminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_x);
    H5Dclose(h_halo_dat);

    /* Open and read the Y-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Ycminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_y);
    H5Dclose(h_halo_dat);

    /* Open and read the Z-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Zcminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_z);
    H5Dclose(h_halo_dat);

    /* Read out the X-velocities */
    h_halo_dat = H5Dopen2(h_halo_file, "VXcminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_vx);
    H5Dclose(h_halo_dat);

    /* Open and read the Y-velocities */
    h_halo_dat = H5Dopen2(h_halo_file, "VYcminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_vy);
    H5Dclose(h_halo_dat);

    /* Open and read the Z-velocities */
    h_halo_dat = H5Dopen2(h_halo_file, "VZcminpot", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_vz);
    H5Dclose(h_halo_dat);


    /* Open and read the hostHaloID */
    h_halo_dat = H5Dopen2(h_halo_file, "hostHaloID", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, hostHaloId);
    H5Dclose(h_halo_dat);

    /* Close the halos file */
    H5Fclose(h_halo_file);

    /* The size of the density grid that we will create */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;
    
    /* Hard coded unit conversions from velociraptor (todo...) */
    const double redshift = 0.0;
    const double vel_conversion_factor = 9.7846194238e2; // km/s to Mpc/Gyr
    
    printf("\n");
    printf("Using N = %d, BoxLen = %g\n", N, boxlen);
    printf("Using redshift z = %f\n", redshift);
    printf("Using velocity conversion factor = %e\n", vel_conversion_factor);
    
    /* The halos under consideration */
    const double M_min = pars.HaloMinMass;
    const double M_max = pars.HaloMaxMass;

    message(rank, "Including halos with M in (%e, %e) U_M.\n", M_min, M_max);
    printf("\n");

    /* We will read the matter velocity fields into these arrays */
    double *vm_x = NULL; //matter x-velocity
    double *vm_y = NULL; //matter y-velocity
    double *vm_z = NULL; //matter z-velocity
    
    /* Read the matter velocity grids */
    char letters[3] = {'x', 'y', 'z'};
    double *grids_m[3] = {vm_x, vm_y, vm_z};
    for (int i=0; i<3; i++) {
        /* Input grid dimensions */
        int read_N;
        double read_boxlen;
        
        /* Filename of velocity input grid */
        char read_fname[50];
        sprintf(read_fname, "%s_%c.hdf5", pars.InputFilename2, letters[i]);
        printf("Reading input array '%s'.\n", read_fname);
        
        /* Read the grid */
        readFieldFile(&grids_m[i], &read_N, &read_boxlen, read_fname);
        
        if (read_N != N || fabs(read_boxlen - boxlen)/boxlen > 1e-5) {
            printf("Error: input dimensions (N,boxlen) do not match parameter.\n");
            exit(1);
        }
    }   
        
    /* Allocate power spectrum arrays for the bootstrap errors */
    int bins = pars.PowerSpectrumBins;
    int num_samples = 8;
    double *bootstrap_ks = calloc(bins, sizeof(double));
    double *bootstrap_Pks = calloc(bins * num_samples, sizeof(double));
    double *bootstrap_obs = calloc(bins, sizeof(double));
    double *reconstructed_Pks = calloc(bins * num_samples, sizeof(double));
    double *matter_self_Pks = calloc(bins * num_samples, sizeof(double));
    double *halo_self_Pks = calloc(bins * num_samples, sizeof(double));
    double mean_total_weight = 0;
    double mean_total_mass = 0;
    
    printf("\n");
    printf("Computing bootstrapped errors in the empirical power spectrum.\n");
    
    /* Bootstrap errors in the empirical power spectrum */
    for (int ITER = 0; ITER < num_samples; ITER++) {
        printf("Iteration %03d/%03d]\n", ITER, num_samples);

        /* Allocate halo momentum grids */
        double *box_px = calloc((long long)N*N*N, sizeof(double));
        double *box_py = calloc((long long)N*N*N, sizeof(double));
        double *box_pz = calloc((long long)N*N*N, sizeof(double));
        
        /* Allocate a grid for the halo overdensity */
        double *delta_h = calloc((long long)N*N*N, sizeof(double));
        
        /* Counters for halo masses and weights (1 or 0) */
        double total_mass = 0;
        double total_weight = 0;

        double grid_cell_vol = boxlen*boxlen*boxlen / ((long long)N*N*N);

        /* Assign the halos to the grid with CIC */
        for (int l=0; l<halo_num; l++) {
            double X = halo_x[l] / (boxlen/N) * (1.0 + redshift);
            double Y = halo_y[l] / (boxlen/N) * (1.0 + redshift);
            double Z = halo_z[l] / (boxlen/N) * (1.0 + redshift);
            double M = halo_M[l];

            double V_X = halo_vx[l] / vel_conversion_factor;
            double V_Y = halo_vy[l] / vel_conversion_factor;
            double V_Z = halo_vz[l] / vel_conversion_factor;

            double W; //weight used in the CIC assignment
            if (M > M_min && M < M_max) {
                W = 1.0;
            } else {
                W = 0.0;
            }

            if (W == 0)
            continue;
            
            /* Randomly select halos for the bootstrap */
            if (rand()%num_samples > 0) continue;

            total_mass += M;
            total_weight += W;

            int iX = (int) floor(X);
            int iY = (int) floor(Y);
            int iZ = (int) floor(Z);

            double shift = 0;
            
            //The search window with respect to the top-left-upper corner
    		int lookLftX = (int) floor((X-iX) - 1.5 + shift);
    		int lookRgtX = (int) floor((X-iX) + 1.5 + shift);
    		int lookLftY = (int) floor((Y-iY) - 1.5 + shift);
    		int lookRgtY = (int) floor((Y-iY) + 1.5 + shift);
    		int lookLftZ = (int) floor((Z-iZ) - 1.5 + shift);
    		int lookRgtZ = (int) floor((Z-iZ) + 1.5 + shift);

            //Do the mass assignment
    		for (int x=lookLftX; x<=lookRgtX; x++) {
    			for (int y=lookLftY; y<=lookRgtY; y++) {
    				for (int z=lookLftZ; z<=lookRgtZ; z++) {
                        double xx = fabs(X - (iX+x+shift));
                        double yy = fabs(Y - (iY+y+shift));
                        double zz = fabs(Z - (iZ+z+shift));

                        double part_x = xx < 0.5 ? (0.75-xx*xx)
                                                : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
        				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                                : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
        				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                                : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);
                                                
                        delta_h[row_major(iX+x, iY+y, iZ+z, N)] += W/grid_cell_vol * (part_x*part_y*part_z);
                        box_px[row_major(iX+x, iY+y, iZ+z, N)] += V_X*W/grid_cell_vol * (part_x*part_y*part_z);
                        box_py[row_major(iX+x, iY+y, iZ+z, N)] += V_Y*W/grid_cell_vol * (part_x*part_y*part_z);
                        box_pz[row_major(iX+x, iY+y, iZ+z, N)] += V_Z*W/grid_cell_vol * (part_x*part_y*part_z);
    				}
    			}
    		}
        }
        
        /* Average weight */
        double avg_density = total_weight / (boxlen*boxlen*boxlen);
        
        /* Update the halo count */
        mean_total_weight += total_weight / num_samples;
        mean_total_mass += total_mass / num_samples;
        
        /* Convert to halo number overdensity: delta_h */
        for (int i=0; i<(long long)N*N*N; i++) {
             delta_h[i] = (delta_h[i] - avg_density) / avg_density;
        }
        /* Convert to halo momentum number density: (1+delta_h) v_h */
        for (int i=0; i<(long long)N*N*N; i++) {
             box_px[i] = box_px[i] / avg_density;
             box_py[i] = box_py[i] / avg_density;
             box_pz[i] = box_pz[i] / avg_density;
        }
        
        /* Compute the empirical power spectrum along each dimension */
        double *grids_h[3] = {box_px, box_py, box_pz};
        for (int dim = 0; dim < 3; dim++) {
            /* Allocate 3D real arrays */
            double *ph_i = (double*) fftw_malloc((long long)N*N*N*sizeof(double));
            double *vm_i = (double*) fftw_malloc((long long)N*N*N*sizeof(double));
            
            /* Copy the correct data */
            memcpy(ph_i, grids_h[dim], (long long)N*N*N*sizeof(double));
            memcpy(vm_i, grids_m[dim], (long long)N*N*N*sizeof(double));
            
            /* Allocate 3D complex arrays */
            fftw_complex *f_ph_i = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
            fftw_complex *f_vm_i = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
            
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
            double *k_in_bins = malloc(bins * sizeof(double));
            double *power_in_bins = malloc(bins * sizeof(double));
            int *obs_in_bins = calloc(bins, sizeof(int));
            
            /* Calculate the power spectrum */
            calc_cross_powerspec(N, boxlen, f_ph_i, f_vm_i, bins, k_in_bins, power_in_bins, obs_in_bins);
            
            /* Add the data (note that we add x + y + z) */
            for (int i=0; i<bins; i++) {
                bootstrap_ks[i] = k_in_bins[i]; //the same everytime
                bootstrap_obs[i] = obs_in_bins[i]; //the same everytime
                bootstrap_Pks[ITER * bins + i] += power_in_bins[i];
            }
            
            /* Calculate the halo autopower spectrum */
            calc_cross_powerspec(N, boxlen, f_ph_i, f_ph_i, bins, k_in_bins, power_in_bins, obs_in_bins);
            
            /* Add the data (note that we add x + y + z) */
            for (int i=0; i<bins; i++) {
                halo_self_Pks[ITER * bins + i] += power_in_bins[i];
            }
            
            /* Calculate the matter autopower spectrum */
            calc_cross_powerspec(N, boxlen, f_vm_i, f_vm_i, bins, k_in_bins, power_in_bins, obs_in_bins);
            
            /* Add the data (note that we add x + y + z) */
            for (int i=0; i<bins; i++) {
                matter_self_Pks[ITER * bins + i] += power_in_bins[i];
            }

            /* Clean up the grids */
            free(f_ph_i);
            free(f_vm_i);
            free(ph_i);
            free(vm_i);
        }
                
        /* Compute the global S power spectrum */
        for (int dim = 0; dim < 3; dim++) {
            /* Allocate 3D real arrays */
            double *vm_i = (double*) fftw_malloc((long long)N*N*N*sizeof(double));

            /* Copy the correct data */
            memcpy(vm_i, grids_m[dim], (long long)N*N*N*sizeof(double));

            /* Allocate 3D complex arrays */
            fftw_complex *f_vm_i = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
            fftw_complex *f_dhvm_i = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

            /* Create FFT plans */
            fftw_plan r2c_1 = fftw_plan_dft_r2c_3d(N, N, N, vm_i, f_vm_i, FFTW_ESTIMATE);
            fftw_plan c2r_1 = fftw_plan_dft_c2r_3d(N, N, N, f_dhvm_i, vm_i, FFTW_ESTIMATE);
            fftw_plan r2c_2 = fftw_plan_dft_r2c_3d(N, N, N, vm_i, f_dhvm_i, FFTW_ESTIMATE);

            /* Execute FFT and normalize */
            fft_execute(r2c_1);
            fft_normalize_r2c(f_vm_i, N, boxlen);
            fftw_destroy_plan(r2c_1);

            /* Copy over the data into the second complex array */
            memcpy(f_dhvm_i, f_vm_i, (long long)N*N*(N/2+1)*sizeof(fftw_complex));

            /* Execute reverse FFT and normalize */
            fft_execute(c2r_1);
            fft_normalize_c2r(vm_i, N, boxlen);
            fftw_destroy_plan(c2r_1);

            /* Multiply by the halo overdensity */
            for (int j=0; j<(long long)N*N*N; j++) {
                vm_i[j] *= delta_h[j];
            }

            /* Execute FFT and normalize */
            fft_execute(r2c_2);
            fft_normalize_r2c(f_dhvm_i, N, boxlen);
            fftw_destroy_plan(r2c_2);

            /* Allocate power spectrum arrays */
            double *k_in_bins = malloc(bins * sizeof(double));
            double *power_in_bins_1 = malloc(bins * sizeof(double));
            double *power_in_bins_2 = malloc(bins * sizeof(double));
            int *obs_in_bins = calloc(bins, sizeof(int));

            /* Compute power spectra */
            calc_cross_powerspec(N, boxlen, f_vm_i, f_vm_i, bins, k_in_bins, power_in_bins_1, obs_in_bins);
            calc_cross_powerspec(N, boxlen, f_dhvm_i, f_vm_i, bins, k_in_bins, power_in_bins_2, obs_in_bins);

            /* Add the data */
            for (int i=0; i<bins; i++) {
                double Pk = power_in_bins_1[i] + power_in_bins_2[i];
                reconstructed_Pks[ITER * bins + i] += Pk;
            }
            
            /* Free the grid */
            free(vm_i);
            free(f_vm_i);
            free(f_dhvm_i);
        }
        
        /* Free the halo overdensity grid */
        free(delta_h);
        
        /* Free the halo momentum boxes */
        free(box_px);
        free(box_py);
        free(box_pz);
    }
    
    printf("\n");
    printf("Mean total weight: %e\n", mean_total_weight);
    printf("Mean total mass: %e\n", mean_total_mass);

    /* Allocate arrays for mean and variance of the power spectra */
    double *bootstrap_Pk_mean = calloc(bins, sizeof(double));
    double *bootstrap_Pk_var = calloc(bins, sizeof(double));
    double *reconstructed_Pk_mean = calloc(bins, sizeof(double));
    double *reconstructed_Pk_var = calloc(bins, sizeof(double));
    double *bias_mean = calloc(bins, sizeof(double));
    double *bias_var = calloc(bins, sizeof(double));
    double *halo_self_Pk_mean = calloc(bins, sizeof(double));
    double *matter_self_Pk_mean = calloc(bins, sizeof(double));
    double *correlation_mean = calloc(bins, sizeof(double));
    double *correlation_var = calloc(bins, sizeof(double));
    
    /* Compute the mean bootstrapped power spectrum */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            bootstrap_Pk_mean[i] += bootstrap_Pks[j * bins + i] / num_samples;
            halo_self_Pk_mean[i] += halo_self_Pks[j * bins + i] / num_samples;
            matter_self_Pk_mean[i] += matter_self_Pks[j * bins + i] / num_samples;
            correlation_mean[i] += bootstrap_Pks[j * bins + i] / sqrt(halo_self_Pks[j * bins + i] * matter_self_Pks[j * bins + i]) / num_samples;
        }
    }
    
    /* Compute the variance of the bootstrapped power spectrum */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            double dPk = (bootstrap_Pks[j * bins + i] - bootstrap_Pk_mean[i]);
            bootstrap_Pk_var[i] += (dPk * dPk) / (num_samples - 1.0);
            double dr = bootstrap_Pks[j * bins + i] / sqrt(halo_self_Pks[j * bins + i] * matter_self_Pks[j * bins + i]) - correlation_mean[i];
            correlation_var[i] += (dr * dr) / (num_samples - 1.0);
        }
    }
    
    /* Compute the mean reconstructed power spectrum */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            reconstructed_Pk_mean[i] += reconstructed_Pks[j * bins + i] / num_samples;
        }
    }
    
    /* Compute the variance of the bootstrapped power spectrum */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            double dPk = (reconstructed_Pks[j * bins + i] - reconstructed_Pk_mean[i]);
            reconstructed_Pk_var[i] += (dPk * dPk) / (num_samples - 1.0);
        }
    }
    
    /* Compute the mean bias */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            double b = bootstrap_Pks[j * bins + i] / reconstructed_Pks[j * bins + i];
            bias_mean[i] += b / num_samples;
        }
    }
    
    /* Compute the variance of the bias */
    for (int i=0; i<bins; i++) {    
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        for (int j=0; j<num_samples; j++) {
            double b = bootstrap_Pks[j * bins + i] / reconstructed_Pks[j * bins + i];
            double db = b - bias_mean[i];
            bias_var[i] += (db * db) / (num_samples - 1.0);
        }
    }
    
    /* Free the full power spectrum arrays */
    free(reconstructed_Pks);
    free(bootstrap_Pks);
    free(halo_self_Pks);
    free(matter_self_Pks);
    
    /* Print the mean of the bootstrapped and self power spectra and the cross-correlation coefficient */
    printf("\n");
    printf("k Pk_cross_mean Pk_halo_mean Pk_matter_mean correlation_mean correlation_var\n");
    for (int i=0; i<bins; i++) {
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        printf("%e %e %e %e %e %e\n", bootstrap_ks[i], bootstrap_Pk_mean[i], halo_self_Pk_mean[i], matter_self_Pk_mean[i], correlation_mean[i], correlation_var[i]);
    }
    printf("\n");
    
    /* Print the mean and error of the bootstrapped power spectrum */
    printf("\n");
    printf("k Pk_reconstruct_mean Pk_bootstrap_mean Pk_reconstruct_var Pk_bootstrap_var bias_mean bias_var\n");
    for (int i=0; i<bins; i++) {
        if (bootstrap_obs[i] <= 1) continue; //skip (nearly) empty bins
        printf("%e %e %e %e %e %e %e\n", bootstrap_ks[i], reconstructed_Pk_mean[i], bootstrap_Pk_mean[i], reconstructed_Pk_var[i], bootstrap_Pk_var[i], bias_mean[i], bias_var[i]);
    }
    printf("\n");
    
    /* Free remaining arrays */
    free(bootstrap_Pk_mean);
    free(bootstrap_Pk_var);
    free(reconstructed_Pk_mean);
    free(reconstructed_Pk_var);
    free(halo_self_Pk_mean);
    free(matter_self_Pk_mean);
    free(bias_mean);
    free(bias_var);
    free(bootstrap_ks);
    free(bootstrap_obs);
    free(correlation_mean);
    free(correlation_var);

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);

    /* Timer */
    gettimeofday(&stop, NULL);
    long unsigned microsec = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\nTime elapsed: %.3f ms\n", microsec/1000.);

}
