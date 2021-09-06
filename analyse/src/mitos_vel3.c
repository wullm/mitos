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

    /* Read parameter file */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);

    /* Option to override the input filename by specifying command line option */
    if (argc > 2) {
        const char *input_filename = argv[2];
        strcpy(pars.InputFilename, input_filename);
    }
    message(rank, "Reading simulation snapshot from: \"%s\".\n", pars.InputFilename);
    
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

    /* Open the file */
    // hid_t h_file = openFile_MPI(MPI_COMM_WORLD, pars.InputFilename);
    hid_t h_file = H5Fopen(pars.InputFilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the physical dimensions of the box */
    double boxlen[3];
    hid_t h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, boxlen);
    H5Aclose(h_attr);
    assert(h_err >= 0);

    boxlen[1] = boxlen[2] = boxlen[0];
    message(rank, "Reading particle type '%s'.\n\n", pars.ImportName);
    message(rank, "BoxSize is %g\n", boxlen[0]);

    /* Read the numbers of particles of each type */
    hsize_t numer_of_types;
    h_attr = H5Aopen(h_grp, "NumPart_Total", H5P_DEFAULT);
    hid_t h_atspace = H5Aget_space(h_attr);
    H5Sget_simple_extent_dims(h_atspace, &numer_of_types, NULL);
    H5Sclose(h_atspace);
    H5Aclose(h_attr);

    /* Close the Header group again */
    H5Gclose(h_grp);

    /* Check if the Cosmology group exists */
    hid_t h_status = H5Eset_auto1(NULL, NULL);  //turn off error printing
    h_status = H5Gget_objinfo(h_file, "/Cosmology", 0, NULL);

    /* If the group exists. */
    if (h_status == 0) {
        /* Open the Cosmology group */
        h_grp = H5Gopen(h_file, "Cosmology", H5P_DEFAULT);

        /* Read the redshift attribute */
        double redshift;
        h_attr = H5Aopen(h_grp, "Redshift", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &redshift);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        message(rank, "The redshift was %f\n\n", redshift);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    }

    /* Try to open the desired import group */

    /* The size of the density grid that we will create */
    const int N = pars.GridSize;

    /* Allocate grids */
    double *box_vx = malloc((long long int) N * N * N * sizeof(double));
    double *box_vy = malloc((long long int) N * N * N * sizeof(double));
    double *box_vz = malloc((long long int) N * N * N * sizeof(double));
    double *box_dens = malloc((long long int) N * N * N * sizeof(double));

    for (int i=0; i<(long long)N*N*N; i++) {
      box_vx[i] = 0;
      box_vy[i] = 0;
      box_vz[i] = 0;
      box_dens[i] = 0;
    }

    /* Open the corresponding group */
    h_grp = H5Gopen(h_file, pars.ImportName, H5P_DEFAULT);

    /* Open the coordinates dataset */
    hid_t h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

    /* Find the dataspace (in the file) */
    hid_t h_space = H5Dget_space (h_dat);

    /* Get the dimensions of this dataspace */
    hsize_t dims[2];
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* How many particles do we want per slab? */
    hid_t Npart = dims[0];
    hid_t max_slab_size = pars.SlabSize;
    int slabs = Npart/max_slab_size;
    hid_t counter = 0;

    /* Close the data and memory spaces */
    H5Sclose(h_space);

    /* Close the dataset */
    H5Dclose(h_dat);

    double total_mass = 0; //for this particle type

    int slab_counter = 0;

    for (int k=rank; k<slabs+1; k+=MPI_Rank_Count) {
        /* All slabs have the same number of particles, except possibly the last */
        hid_t slab_size = fmin(Npart - k * max_slab_size, max_slab_size);
        counter += slab_size; //the number of particles read

        /* Define the hyperslab */
        hsize_t slab_dims[2], start[2]; //for 3-vectors
        hsize_t slab_dims_one[1], start_one[1]; //for scalars

        /* Slab dimensions for 3-vectors */
        slab_dims[0] = slab_size;
        slab_dims[1] = 3; //(x,y,z)
        start[0] = k * max_slab_size;
        start[1] = 0; //start with x

        /* Slab dimensions for scalars */
        slab_dims_one[0] = slab_size;
        start_one[0] = k * max_slab_size;

        /* Open the coordinates dataset */
        h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                           NULL, slab_dims, NULL);
        assert(status >= 0);

        /* Create a memory space */
        hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

        /* Create the data array */
        double data[slab_size][3];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);


        /* Open the masses dataset */
        h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        h_space = H5Dget_space (h_dat);

        /* Select the hyperslab */
        status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                            slab_dims_one, NULL);

        /* Create a memory space */
        h_mems = H5Screate_simple(1, slab_dims_one, NULL);

        /* Create the data array */
        double mass_data[slab_size];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         mass_data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);


         /* Open the velocities dataset */
         h_dat = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);

         /* Find the dataspace (in the file) */
         h_space = H5Dget_space (h_dat);

         /* Select the hyperslab */
         status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                      NULL, slab_dims, NULL);
         assert(status >= 0);

         /* Create a memory space */
         h_mems = H5Screate_simple(2, slab_dims, NULL);

         /* Create the data array */
         double velocities_data[slab_size][3];

         status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                          velocities_data);

         /* Close the memory space */
         H5Sclose(h_mems);

         /* Close the data and memory spaces */
         H5Sclose(h_space);

         /* Close the dataset */
         H5Dclose(h_dat);

        double grid_cell_vol = boxlen[0]*boxlen[1]*boxlen[2] / ((long long)N*N*N);

        /* Assign the particles to the grid with CIC */
        for (int l=0; l<slab_size; l++) {
            double X = data[l][0] / (boxlen[0]/N);
            double Y = data[l][1] / (boxlen[1]/N);
            double Z = data[l][2] / (boxlen[2]/N);

            double V_X = velocities_data[l][0];
            double V_Y = velocities_data[l][1];
            double V_Z = velocities_data[l][2];

            double M = mass_data[l];
            total_mass += M;

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

                        box_vx[row_major(iX+x, iY+y, iZ+z, N)] += V_X*M/grid_cell_vol * (part_x*part_y*part_z);
                        box_vy[row_major(iX+x, iY+y, iZ+z, N)] += V_Y*M/grid_cell_vol * (part_x*part_y*part_z);
                        box_vz[row_major(iX+x, iY+y, iZ+z, N)] += V_Z*M/grid_cell_vol * (part_x*part_y*part_z);
                        box_dens[row_major(iX+x, iY+y, iZ+z, N)] += M/grid_cell_vol * (part_x*part_y*part_z);
    				}
    			}
    		}
        }

        printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
        slab_counter++;
    }

    /* Close the group again */
    H5Gclose(h_grp);


    /* Reduce the grids */
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, box_vx, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(box_vx, box_vx, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, box_vy, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(box_vy, box_vy, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, box_vz, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(box_vz, box_vz, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, box_dens, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(box_dens, box_dens, (long long) N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* Reduce the total mass */
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&total_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }


    if (rank == 0) {

        message(rank, "Total mass: %f\n", total_mass);

        /* The average density */
        double avg_density = total_mass / (boxlen[0]*boxlen[1]*boxlen[2]);

        message(rank, "Average density %f\n", avg_density);

        double w = 0.;
        printf("Using equation of state w = %f\n", w);

        /* Turn the density field into an overdensity field */
        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    long long int id = row_major(x, y, z, N);
                    /* Overdensity */
                    box_dens[id] = (box_dens[id] - avg_density)/avg_density;

                    /* Velocity fields */
                    box_vx[id] = box_vx[id] / avg_density / (1. + w) / (1.0 + box_dens[id]);
                    box_vy[id] = box_vy[id] / avg_density / (1. + w) / (1.0 + box_dens[id]);
                    box_vz[id] = box_vz[id] / avg_density / (1. + w) / (1.0 + box_dens[id]);
                }
            }
        }

        /* Find a particle type with a matching ExportName */
        struct particle_type *tp;
        char found = 0;
        for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
            struct particle_type *ptype = types + pti;
            const char *ExportName = ptype->ExportName;

            if (strcmp(ExportName, pars.ImportName) == 0) {
                tp = ptype;
                found = 1;
            }
        }

        if (disable_write == 0) {
            char dirs[3] = "xyz";
            for (int dir=0; dir<3; dir++) {
                char box_fname[40];
                if (!found) {
                    sprintf(box_fname, "velocity_%s_%c.hdf5", pars.ImportName, dirs[dir]);
                } else {
                    sprintf(box_fname, "velocity_%s_%c.hdf5", tp->Identifier, dirs[dir]);
                }
                if (dir==0)
                    writeFieldFileCompressed(box_vx, N, boxlen[0], box_fname, pars.LossyScaleDigits);
                else if (dir==1)
                    writeFieldFileCompressed(box_vy, N, boxlen[0], box_fname, pars.LossyScaleDigits);
                else
                    writeFieldFileCompressed(box_vz, N, boxlen[0], box_fname, pars.LossyScaleDigits);

                message(rank, "Velocity grid exported to %s.\n", box_fname);
            }

          char box_fname[40];
          if (!found) {
            sprintf(box_fname, "density_%s.hdf5", pars.ImportName);
          } else {
            sprintf(box_fname, "density_%s.hdf5", tp->Identifier);
          }
          writeFieldFileCompressed(box_dens, N, boxlen[0], box_fname, pars.LossyScaleDigits);
          message(rank, "Density grid exported to %s.\n", box_fname);
      }
    }

    if (rank == 0) {

        printf("\n");

        printf("Doing ffts\n");

        /* Transform to momentum space */
        fftw_complex *fbox_x = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_plan r2c_x = fftw_plan_dft_r2c_3d(N, N, N, box_vx, fbox_x, FFTW_ESTIMATE);
        fft_execute(r2c_x);
        fft_normalize_r2c(fbox_x,N,boxlen[0]);

        fftw_complex *fbox_y = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_plan r2c_y = fftw_plan_dft_r2c_3d(N, N, N, box_vy, fbox_y, FFTW_ESTIMATE);
        fft_execute(r2c_y);
        fft_normalize_r2c(fbox_y,N,boxlen[0]);

        fftw_complex *fbox_z = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_plan r2c_z = fftw_plan_dft_r2c_3d(N, N, N, box_vz, fbox_z, FFTW_ESTIMATE);
        fft_execute(r2c_z);
        fft_normalize_r2c(fbox_z,N,boxlen[0]);

        //printf("Undoing window function\n");

        /* Undo the TSC window function */
        //undoTSCWindow(fbox_x, N, boxlen[0]);
        //undoTSCWindow(fbox_y, N, boxlen[0]);
        //undoTSCWindow(fbox_z, N, boxlen[0]);

        printf("Computing derivatives\n");

        fft_apply_kernel(fbox_x, fbox_x, N, boxlen[0], kernel_dx, NULL);
        fft_apply_kernel(fbox_y, fbox_y, N, boxlen[0], kernel_dy, NULL);
        fft_apply_kernel(fbox_z, fbox_z, N, boxlen[0], kernel_dz, NULL);

        printf("Computing power spectrum of theta\n");

        fftw_complex *fbox = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

        for (int k=0; k<(long long)N*N*(N/2+1); k++) {
          fbox[k][0] = fbox_x[k][0] + fbox_y[k][0] + fbox_z[k][0];
          fbox[k][1] = fbox_x[k][1] + fbox_y[k][1] + fbox_z[k][1];
        }

        int bins = 50;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));


         /* Reset the bins */
         for (int i=0; i<bins; i++) {
             k_in_bins[i] = 0;
             power_in_bins[i] = 0;
             obs_in_bins[i] = 0;
         }

         /* Cross spectrum of cdm and ncdm */
         calc_cross_powerspec(N, boxlen[0], fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

         /* Print the results */
         printf("k P_measured(k) observations\n");
         for (int i=0; i<bins; i++) {
             if (obs_in_bins[i] == 0) continue; //skip empty bins

             /* The power we observe */
             double k = k_in_bins[i];
             double Pk = power_in_bins[i];
             int obs = obs_in_bins[i];

             printf("%f %e %d\n", k, Pk, obs);
         }

         printf("\n");

        /* Transform back */
        double *box = malloc((long long int) N * N * N * sizeof(double));
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(c2r);
        fft_normalize_c2r(box,N,boxlen[0]);

        /* Export the density box for testing purposes */
        if (disable_write == 0) {
            /* Find a particle type with a matching ExportName */
            struct particle_type *tp;
            char found = 0;
            for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
                struct particle_type *ptype = types + pti;
                const char *ExportName = ptype->ExportName;

                if (strcmp(ExportName, pars.ImportName) == 0) {
                    tp = ptype;
                    found = 1;
                }
            }

            char box_fname[40];
            if (!found) {
              sprintf(box_fname, "theta_%s.hdf5", pars.ImportName);
            } else {
              sprintf(box_fname, "theta_%s.hdf5", tp->Identifier);
            }

            writeFieldFileCompressed(box, N, boxlen[0], box_fname, pars.LossyScaleDigits);
            printf("Flux density grid exported to %s.\n", box_fname);
        }
        
        free(fbox_x);
        free(fbox_y);
        free(fbox_z);
        free(fbox);
        fftw_free(box);
    }
    
    fftw_free(box_vx);
    fftw_free(box_vy);
    fftw_free(box_vz);
    fftw_free(box_dens);

    /* Close the HDF5 file */
    H5Fclose(h_file);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
