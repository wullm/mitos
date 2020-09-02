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

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);

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

    /* Close the halos file */
    H5Fclose(h_halo_file);

    /* Allocate memory for the profiles */
    int num_bins = 25;
    double max_radius = 2.50;
    double *profiles = calloc(num_bins * halo_num, sizeof(double));


    /* Open the mesh file */
    int box_N;
    double boxlen;
    double *box;
    readFieldFile(&box, &box_N, &boxlen, pars.InputFilename);

    /* For each halo */
    for (int i=0; i<halo_num; i++) {

        /* Coordinates of the halo centre */
        double cx = halo_x[i];
        double cy = halo_y[i];
        double cz = halo_z[i];

        /* For each bin */
        for (int j=0; j<num_bins; j++) {

            /* The radius of this bin */
            double r = max_radius / num_bins * j;
            /* Sample K points at distance r */
            int K = 100;

            /* For each point */
            for (int k=0; k<K; k++) {
                /* Generate a random point on the unit sphere using Gaussians */
                double nx = sampleNorm(&seed);
                double ny = sampleNorm(&seed);
                double nz = sampleNorm(&seed);

                /* And normalize */
                double length = hypot(nx, hypot(ny, nz));
                if (length > 0) {
                    nx /= length;
                    ny /= length;
                    nz /= length;
                }

                /* The point in question */
                double px = cx + r * nx;
                double py = cy + r * ny;
                double pz = cz + r * nz;

                /* Fetch the overdensity at this point */
                double d = gridCIC(box, box_N, boxlen, px, py, pz);

                /* Add the density */
                profiles[i * num_bins + j] += d/K;
            }

        }
    }

    printf("%f %f %f\n", boxlen, boxlen, boxlen);

    /* Print some profiles */
    for (int h_id = 0; h_id < halo_num; h_id += 1000) {
        for (int i=0; i < num_bins; i++) {
            printf("%e ", profiles[h_id * num_bins + i]);
        }
        printf("\n");
    }

    if (rank == 0) {
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
        h_data = H5Dcreate(h_exgrp, "Xcminpot", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_x);
        H5Dclose(h_data);

        /* Export X-coordinates */
        h_data = H5Dcreate(h_exgrp, "Ycminpot", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_y);
        H5Dclose(h_data);

        /* Export Z-coordinates */
        h_data = H5Dcreate(h_exgrp, "Zcminpot", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_sspace, h_sspace, H5P_DEFAULT, halo_z);
        H5Dclose(h_data);

        /* Close the scalar dataspace */
        H5Sclose(h_sspace);

        /* Close the Halos group */
        H5Gclose(h_exgrp);

        /* Close the profiles file */
        H5Fclose(out_file);

    }

    /* Free memory for the halo stats */
    free(halo_M);
    free(halo_x);
    free(halo_y);
    free(halo_z);
    free(profiles);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
