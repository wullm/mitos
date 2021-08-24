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
#include <hdf5.h>
#include <assert.h>
#include <math.h>

#include "../../include/mitos.h"

/* Used for linked list */
struct id_link {
    int id;
    struct id_link *next;
};

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
    message(rank, "The parameter file is %s\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);

    /* Option to override the input filename by specifying command line option */
    if (argc > 2) {
        const char *input_filename = argv[2];
        strcpy(pars.HaloInputFilename, input_filename);
    }
    message(rank, "Reading halos from '%s'.\n", pars.HaloInputFilename);

    /* Option to disable output writing */
    int disable_write = 0;
    if (argc > 3) {
        disable_write = 1;
        message(rank, "Not writing output. Only computing power spectrum.\n");
    }

    /* Be verbose about the use of lossy output filters */
    if (pars.LossyScaleDigits > 0) {
        message(rank, "Using lossy output filter (keeping %d decimals).\n",
                pars.LossyScaleDigits);
    }

    /* Open the Halos file */
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

    /* The halos under consideration */
    const double M_min = pars.HaloMinMass;
    const double M_max = pars.HaloMaxMass;

    message(rank, "Including halos with M in (%e, %e) U_M.\n", M_min, M_max);



    /* Allocate grids */
    double *box = fftw_alloc_real(N * N * N);
    double *box_px = fftw_alloc_real(N * N * N);
    double *box_py = fftw_alloc_real(N * N * N);
    double *box_pz = fftw_alloc_real(N * N * N);

    double total_mass = 0;
    double total_weight = 0;

    double grid_cell_vol = boxlen*boxlen*boxlen / (N*N*N);

    //double redz = 1.274318;
    double redz = 0.0;

    printf("The used redshift was %f\n", redz);

    /* Assign the halos to the grid with CIC */
    for (int l=0; l<halo_num; l++) {
        double X = halo_x[l] / (boxlen/N) * (1+redz);
        double Y = halo_y[l] / (boxlen/N) * (1+redz);
        double Z = halo_z[l] / (boxlen/N) * (1+redz);
        double M = halo_M[l]; // / (boxlen/N);

        double V_X = halo_vx[l];
        double V_Y = halo_vy[l];
        double V_Z = halo_vz[l];

        //long long hid = hostHaloId[l];
        //if (hid>-1) continue;

        double W; //weight used in the CIC assignment
        if (M > M_min && M < M_max) {
            W = 1.0;
        } else {
            W = 0.0;
        }

        if (W == 0)
        continue;

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

                    box[row_major(iX+x, iY+y, iZ+z, N)] += W/grid_cell_vol * (part_x*part_y*part_z);
                    box_px[row_major(iX+x, iY+y, iZ+z, N)] += V_X*W/grid_cell_vol * (part_x*part_y*part_z);
                    box_py[row_major(iX+x, iY+y, iZ+z, N)] += V_Y*W/grid_cell_vol * (part_x*part_y*part_z);
                    box_pz[row_major(iX+x, iY+y, iZ+z, N)] += V_Z*W/grid_cell_vol * (part_x*part_y*part_z);
				}
			}
		}
    }
    
    /* Average weight */
    double avg_density = total_weight / (boxlen*boxlen*boxlen);
    
    printf("Using BoxLen = %g\n", boxlen);
    printf("Average density = %g\n", avg_density);

    /* Convert to halo momentum number density: (1+delta_h) v_h */
    for (int i=0; i<N*N*N; i++) {
         box_px[i] = box_px[i] / avg_density;
         box_py[i] = box_py[i] / avg_density;
         box_pz[i] = box_pz[i] / avg_density;
    }

    if (disable_write == 0) {
        writeFieldFileCompressed(box_px, N, boxlen, "momentum_halos_x.hdf5", pars.LossyScaleDigits);
        message(rank, "Halo momentum grid exported to %s.\n", "momentum_halos_x.hdf5");
        writeFieldFileCompressed(box_py, N, boxlen, "momentum_halos_y.hdf5", pars.LossyScaleDigits);
        message(rank, "Halo momentum grid exported to %s.\n", "momentum_halos_y.hdf5");
        writeFieldFileCompressed(box_pz, N, boxlen, "momentum_halos_z.hdf5", pars.LossyScaleDigits);
        message(rank, "Halo momentum grid exported to %s.\n", "momentum_halos_z.hdf5");
        writeFieldFileCompressed(box, N, boxlen, "density_halos.hdf5", pars.LossyScaleDigits);
        message(rank, "Halo density grid exported to %s.\n", "density_halos.hdf5");
    }

    message(rank, "Total mass = %e\n", total_mass);
    message(rank, "Total weight = %e\n", total_weight);

    /* Prepare for computing the power spectrum */
    int bins = 50;
    double *k_in_bins = malloc(bins * sizeof(double));
    double *power_in_bins = malloc(bins * sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));

    /* Transform to momentum space */
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fbox,N,boxlen);

    /* Undo the TSC window function */
    undoTSCWindow(fbox, N, boxlen);

    calc_cross_powerspec(N, boxlen, fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

    /* Check that it is right */
    printf("\n");
    printf("Power spectrum:\n");
    printf("k P_measured(k) observations\n");
    for (int i=0; i<bins; i++) {
        if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

        /* The power we observe */
        double k = k_in_bins[i];
        double Pk = power_in_bins[i];
        int obs = obs_in_bins[i];

        printf("%f %e %d\n", k, Pk, obs);
    }

    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);
    fftw_free(fbox);

    printf("\n");


    /* Free memory for the halo stats */
    free(halo_M);
    free(halo_x);
    free(halo_y);
    free(halo_z);
    free(halo_vx);
    free(halo_vy);
    free(halo_vz);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
