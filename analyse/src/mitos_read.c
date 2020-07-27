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

const char *fname;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printf("The parameter file is %s\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);

    printf("Reading simulation snapshot for: \"%s\".\n", pars.Name);

    /* Open the file */
    hid_t h_file = H5Fopen(pars.InputFilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the physical dimensions of the box */
    double boxlen[3];
    hid_t h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, boxlen);
    H5Aclose(h_attr);
    assert(h_err >= 0);

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

        printf("The redshift was %f\n\n", redshift);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    }

    /* We will copy the Fourier transform CDM density into here later */
    fftw_complex *fbox_c = NULL;

    /* Try to open each particle group */
    for (int i=0; i<pars.NumParticleTypes; i++) {
        struct particle_type tp = types[i];
        printf("Found particle type\t %s\n", tp.ExportName);

        /* The size of the density grid that we will create */
        const int N = pars.GridSize;

        /* We will store the density grid in here */
        double *rho_box = calloc(N*N*N, sizeof(double));
        double *rho_interlaced_box = calloc(N*N*N, sizeof(double));

        /* Open the corresponding group */
        hid_t h_grp = H5Gopen(h_file, tp.ExportName, H5P_DEFAULT);

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

        for (int k=0; k<slabs+1; k++) {
            /* All slabs have the same number of particles, except possibly the last */
            hid_t slab_size = fmin(Npart - counter, max_slab_size);
            counter += slab_size; //the number of particles read

            /* Define the hyperslab */
            hsize_t slab_dims[2], start[2]; //for 3-vectors
            hsize_t slab_dims_one[1], start_one[1]; //for scalars

            /* Slab dimensions for 3-vectors */
            slab_dims[0] = slab_size;
            slab_dims[1] = 3; //(x,y,z)
            start[0] = counter - slab_size;
            start[1] = 0; //start with x

            /* Slab dimensions for scalars */
            slab_dims_one[0] = slab_size;
            start_one[0] = counter - slab_size;

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



            double grid_cell_vol = boxlen[0]*boxlen[1]*boxlen[2] / (N*N*N);

            /* Assign the particles to the grid with CIC */
            for (int l=0; l<slab_size; l++) {
                double X = data[l][0] / (boxlen[0]/N);
                double Y = data[l][1] / (boxlen[1]/N);
                double Z = data[l][2] / (boxlen[2]/N);

                double V_X = velocities_data[l][0];
                double V_Y = velocities_data[l][1];
                double V_Z = velocities_data[l][2];

                /* Unused variables */
                (void) V_X;
                (void) V_Y;
                (void) V_Z;

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

                            rho_box[row_major(iX+x, iY+y, iZ+z, N)] += M/grid_cell_vol * (part_x*part_y*part_z);
        				}
        			}
        		}
            }

            printf("%d) Read %ld particles\n", slab_counter, slab_size);
            slab_counter++;
        }

        /* Close the group again */
        H5Gclose(h_grp);

        printf("Total mass: %f\n", total_mass);

        /* The average density */
        double avg_density = total_mass / (boxlen[0]*boxlen[1]*boxlen[2]);

        printf("Average density %f\n", avg_density);

        if (strcmp(tp.Identifier, "ncdm") == 0) {
            avg_density = 0.179075;
            printf("Reset avg_density to %f\n", avg_density);
        }

        /* Turn the density field into an overdensity field */
        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    int id = row_major(x, y, z, N);
                    if (strcmp(tp.Identifier, "ncdm") == 0) {
                        rho_box[id] = rho_box[id]/avg_density;
                    } else {
                        rho_box[id] = (rho_box[id] - avg_density)/avg_density;
                    }
                }
            }
        }

        // readGRF_inPlace_H5(rho_box, "output/density_cdm.hdf5");

        int bins = 50;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));

        /* Transform to momentum space */
        fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, rho_box, fbox, FFTW_ESTIMATE);
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, rho_box, FFTW_ESTIMATE);
        fft_execute(r2c);
    	fft_normalize_r2c(fbox,N,boxlen[0]);

        /* Undo the TSC window function */
        undoTSCWindow(fbox, N, boxlen[0]);

        if (strcmp(tp.Identifier, "cdm") == 0) {
            fbox_c = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
            memcpy(fbox_c, fbox, N*N*(N/2+1)*sizeof(fftw_complex));
        }

        calc_cross_powerspec(N, boxlen[0], fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

        /* Check that it is right */
        printf("\n");
        printf("Example power spectrum:\n");
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


        if (strcmp(tp.Identifier, "ncdm") == 0) {
            /* Calculate the cross spectrum */
            printf("Doing the (c,nu) cross spectrum.\n");

            /* Reset the bins */
            for (int i=0; i<bins; i++) {
                k_in_bins[i] = 0;
                power_in_bins[i] = 0;
                obs_in_bins[i] = 0;
            }

            /* Cross spectrum of cdm and ncdm */
            calc_cross_powerspec(N, boxlen[0], fbox_c, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

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

            free(k_in_bins);
            free(power_in_bins);
            free(obs_in_bins);
        }


        /* Transform back */
        fft_execute(c2r);
    	fft_normalize_c2r(rho_box,N,boxlen[0]);

        /* Export the density box for testing purposes */
        char box_fname[40];
        sprintf(box_fname, "density_%s.hdf5", tp.Identifier);
        writeGRF_H5(rho_box, N, boxlen[0], box_fname);
        printf("Density grid exported to %s.\n", box_fname);

        free(rho_box);
        free(rho_interlaced_box);
        fftw_free(fbox);
    }

    /* Close the HDF5 file */
    H5Fclose(h_file);

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
    fftw_free(fbox_c);
}
