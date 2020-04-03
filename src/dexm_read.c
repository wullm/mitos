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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <math.h>

#include "../include/dexm.h"

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
    struct transfer trs;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, fname);
    readTypes(&pars, &types, fname);

    int err;
    err = readTransfers(&pars, &us, &cosmo, &trs);
    if(err) {
        return 0;
    }

    printf("Creating initial conditions for: \"%s\".\n", pars.Name);

    /* Seed the random number generator */
    srand(pars.Seed);

    /* Open the file */
    hid_t h_file = H5Fopen(pars.InputFilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the physical dimensions of the box */
    double boxlen[3];
    hid_t h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, boxlen);
    H5Aclose(h_attr);

    /* Read the numbers of particles of each type */
    hsize_t numer_of_types;
    h_attr = H5Aopen(h_grp, "NumPart_Total", H5P_DEFAULT);
    hid_t h_atspace = H5Aget_space(h_attr);
    H5Sget_simple_extent_dims(h_atspace, &numer_of_types, NULL);
    H5Sclose(h_atspace);
    H5Aclose(h_attr);

    /* Close the Header group again */
    H5Gclose(h_grp);

    /* Try to open each particle group */
    for (int i=0; i<pars.NumParticleTypes; i++) {
        struct particle_type tp = types[i];
        printf("Found particle type\t %s\n", tp.ExportName);

        /* The size of the density grid that we will create */
        const int N = pars.GridSize;

        /* We will store the density grid in here */
        double *rho_box = calloc(N*N*N, sizeof(double));

        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    rho_box[row_major(x,y,z,N)] = 0.0;
                }
            }
        }

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

        for (int k=0; k<slabs+1; k++) {
            /* All slabs have the same number of particles, except possibly the last */
            hid_t slab_size = fmin(Npart - counter, max_slab_size);
            counter += slab_size; //the number of particles read

            /* Define the hyperslab */
            hsize_t slab_dims[2], start[2];

            slab_dims[0] = slab_size;
            slab_dims[1] = 3; //(x,y,z)
            start[0] = counter - slab_size;
            start[1] = 0; //start with x

            /* Select the hyperslab */
            hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start, NULL,
                                               slab_dims, NULL);


            /* Create a memory space */
            hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

            /* Create the data array */
            double data[slab_size][3];

            status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                             data);

            /* Assign the particles to the grid with CIC */
            for (int l=0; l<slab_size; l++) {
                double X = data[l][0] / (boxlen[0]/N);
                double Y = data[l][1] / (boxlen[1]/N);
                double Z = data[l][2] / (boxlen[2]/N);

                // printf("%f %f %f\n", X, Y, Z);

                int iX = (int) floor(X);
                int iY = (int) floor(Y);
                int iZ = (int) floor(Z);

                double shift = 0;

                //The search window with respect to the top-left-upper corner
        		int lookLftX = (int) floor((X-iX) - 1 + shift);
        		int lookRgtX = (int) floor((X-iX) + 1 + shift);
        		int lookLftY = (int) floor((Y-iY) - 1 + shift);
        		int lookRgtY = (int) floor((Y-iY) + 1 + shift);
        		int lookLftZ = (int) floor((Z-iZ) - 1 + shift);
        		int lookRgtZ = (int) floor((Z-iZ) + 1 + shift);

                //Do the mass assignment
        		for (int x=lookLftX; x<=lookRgtX; x++) {
        			for (int y=lookLftY; y<=lookRgtY; y++) {
        				for (int z=lookLftZ; z<=lookRgtZ; z++) {
        					double part_x = fabs(X - (iX+x+shift)) <= 1 ? 1-fabs(X - (iX+x+shift)) : 0;
        					double part_y = fabs(Y - (iY+y+shift)) <= 1 ? 1-fabs(Y - (iY+y+shift)) : 0;
        					double part_z = fabs(Z - (iZ+z+shift)) <= 1 ? 1-fabs(Z - (iZ+z+shift)) : 0;

                            rho_box[row_major(iX+x, iY+y, iZ+z, N)] += 1.0 * (part_x*part_y*part_z);
        				}
        			}
        		}

            }

            /* Close the memory space */
            H5Sclose(h_mems);

            printf("Read %lld particles\n", slab_size);
        }

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        /* Close the group again */
        H5Gclose(h_grp);

        /* Export the density box for testing purposes */
        const char box_fname[40];
        sprintf(box_fname, "density_%s.box", tp.Identifier);
        write_doubles_as_floats(box_fname, rho_box, N*N*N);
        printf("Density grid exported to %s.\n", box_fname);

        free(rho_box);
    }

    /* Close the HDF5 file */
    H5Fclose(h_file);




    /* Clean up */
    cleanTransfers(&trs);
    cleanTypes(&pars, &types);
    cleanParams(&pars);

}
