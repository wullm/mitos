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

    message(rank, "Reading simulation snapshot for: \"%s\".\n", pars.Name);


    /* Open the Halos file */
    message(rank, "Reading halos from '%s'.\n", pars.HaloInputFilename);
    hid_t h_halo_file = openFile_MPI(MPI_COMM_WORLD, pars.HaloInputFilename);

    /* Open the halo masses dataset */
    hid_t h_halo_dat = H5Dopen2(h_halo_file, "Mass_tot", H5P_DEFAULT);

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

    message(rank, "We have %lld halos\n", halo_num);

    /* Read out the masses */
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_M);
    H5Dclose(h_halo_dat);

    /* Read out the X-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Xc", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_x);
    H5Dclose(h_halo_dat);

    /* Open and read the Y-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Yc", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_y);
    H5Dclose(h_halo_dat);

    /* Open and read the Z-coordinates */
    h_halo_dat = H5Dopen2(h_halo_file, "Zc", H5P_DEFAULT);
    H5Dread(h_halo_dat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, halo_z);
    H5Dclose(h_halo_dat);

    /* Close the halos file */
    H5Fclose(h_halo_file);

    /* Allocate memory for the profiles */
    int num_bins = 25;
    double max_radius = 2.50;
    double r2_max = max_radius * max_radius;
    double *profiles = calloc(num_bins * halo_num, sizeof(double));






    /* Open the file */
    hid_t h_file = openFile_MPI(MPI_COMM_WORLD, pars.InputFilename);

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

        message(rank, "The redshift was %f\n\n", redshift);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    }

    /* Try to open the desired import group */


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



    /* The size of the linked list grid */
    const int N = ceil(boxlen[0]/max_radius); //pars.GridSize;
    printf("Reference grid is %d^3.\n", N);

    /* Allocate grids */
    struct id_link *halo_links = malloc(halo_num * sizeof(struct id_link));
    struct id_link **halo_grid = malloc(N * N * N * sizeof(struct id_link*));
    int *halos_in_cell = calloc(N * N * N, sizeof(int));

    /* Insert the halos */
    for (int i=0; i<halo_num; i++) {
        /* Transform physical coordinates to grid coordinates */
        double X = halo_x[i] / (boxlen[0]/N);
        double Y = halo_y[i] / (boxlen[1]/N);
        double Z = halo_z[i] / (boxlen[2]/N);

        /* Integer coordinas of the halo */
        int iX = (int) floor(X);
        int iY = (int) floor(Y);
        int iZ = (int) floor(Z);

        /* ID of the grid cell */
        int id = row_major(iX, iY, iZ, N);

        /* If it is the first halo in this cell */
        if (halos_in_cell[id] == 0) {
            /* Insert in the cell without a link */
            halo_links[i].id = i;
            halo_links[i].next = NULL;
            halo_grid[id] = &halo_links[i];
        } else {
            /* Otherwise, insert in front of the current linked list */
            halo_links[i].id = i;
            halo_links[i].next = halo_grid[id];
            halo_grid[id] = &halo_links[i];
        }

        /* Increment the halos in cell counter */
        halos_in_cell[id]++;
    }

    printf("Reference grid created.\n");

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

        /* Assign the particles to the halo profiles */
        for (int l=0; l<slab_size; l++) {
            double X = data[l][0] / (boxlen[0]/N);
            double Y = data[l][1] / (boxlen[1]/N);
            double Z = data[l][2] / (boxlen[2]/N);

            int iX = (int) floor(X);
            int iY = (int) floor(Y);
            int iZ = (int) floor(Z);

            double M = mass_data[l];
            total_mass += M;

            /* Find halos that are close */
            for (int dx=-1; dx<=1; dx++) {
                for (int dy=-1; dy<=1; dy++) {
                    for (int dz=-1; dz<=1; dz++) {
                        /* Grid cell ID */
                        int grid_id = row_major(iX + dx, iY + dy, iZ + dz, N);
                        /* Number of halos in this cell */
                        int h_max = halos_in_cell[grid_id];

                        /* Next halo */
                        struct id_link *next = halo_grid[grid_id];

                        for (int h=0; h<h_max; h++) {
                            /* Halo id */
                            int h_id = next->id;
                            /* Next halo */
                            next = next->next;

                            /* Determine distance to halo centre */
                            double h_x = halo_x[h_id];
                            double h_y = halo_y[h_id];
                            double h_z = halo_z[h_id];

                            double r2 = (h_x - X)*(h_x - X) + (h_y - Y)*(h_y - Y) + (h_z - Z)*(h_z - Z);

                            if (r2 < r2_max) {
                                /* Put the particle in a bin */
                                int bin = floor(sqrt(r2 / r2_max) * num_bins);
                                profiles[h_id * num_bins + bin] += M;
                            }
                        }

                    }
                }
            }
        }

        printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
        slab_counter++;
    }

    printf("%f %f %f\n", boxlen[0], boxlen[1], boxlen[2]);

    /* Print some profiles */
    for (int h_id = 0; h_id < halo_num; h_id += 1000) {
        for (int i=0; i < num_bins; i++) {
            printf("%e ", profiles[h_id * num_bins + i]);
        }
        printf("\n");
    }

    /* Close the group again */
    H5Gclose(h_grp);


    /* Reduce the grid */
    // if (rank == 0) {
    //     MPI_Reduce(MPI_IN_PLACE, box, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Reduce(box, box, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // }

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

        /* Export the halo profiles */
        char prof_fname[40];
        if (!found) {
            sprintf(prof_fname, "profiles_%s.hdf5", pars.ImportName);
        } else {
            sprintf(prof_fname, "profiles_%s.hdf5", tp->Identifier);
        }

        /* Create the profiles file */
        hid_t out_file = createFile(prof_fname);

        /* Create the Halos group */
        hid_t h_exgrp = H5Gcreate(out_file, "/Halos", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Create dataspace for the profiles */
        const hsize_t crank = 2;
        const hsize_t cdims[2] = {halo_num, num_bins}; //3D space
        hid_t h_cspace = H5Screate_simple(crank, cdims, NULL);

        /* Create the dataset for the profiles */
        hid_t h_data = H5Dcreate(h_exgrp, "Profiles", H5T_NATIVE_DOUBLE, h_cspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Write the data */
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_cspace, h_cspace, H5P_DEFAULT, profiles);

        /* Close the dataset & corresponding dataspace */
        H5Dclose(h_data);
        H5Sclose(h_cspace);

        /* Create dataspace for the masses */
        const hsize_t srank = 1;
        const hsize_t sdims[1] = {halo_num};
        hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

        /* Create the dataset for the masses */
        h_data = H5Dcreate(h_exgrp, "Mass_tot", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Write the data */
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_M);

        /* Close the dataset */
        H5Dclose(h_data);

        /* Export X-coordinates */
        h_data = H5Dcreate(h_exgrp, "Xc", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_x);
        H5Dclose(h_data);

        /* Export X-coordinates */
        h_data = H5Dcreate(h_exgrp, "Yc", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_y);
        H5Dclose(h_data);

        /* Export Z-coordinates */
        h_data = H5Dcreate(h_exgrp, "Zc", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_z);
        H5Dclose(h_data);

        /* Close the scalar dataspace */
        H5Sclose(h_sspace);

        /* Close the Halos group */
        H5Gclose(h_exgrp);

        /* Close the profiles file */
        H5Fclose(out_file);

    }

    /* Free the linked list grid */
    free(halo_links);
    free(halo_grid);
    free(halos_in_cell);

    /* Free memory for the halo stats */
    free(halo_M);
    free(halo_x);
    free(halo_y);
    free(halo_z);
    free(profiles);

    /* Close the HDF5 file */
    H5Fclose(h_file);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
