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

#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>
#include "../include/input_mpi.h"

int readField_MPI(double *data, int N, int NX, int X0, MPI_Comm comm,
                  const char *fname) {

    /* Open the hdf5 file */
    hid_t h_file = openFile_MPI(comm, fname);

    /* Open the Field group */
    hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Get the file dataspace */
    hid_t h_space = H5Dget_space(h_data);

    /* The chunk in question */
    const hsize_t chunk_rank = 3;
    const hsize_t chunk_dims[3] = {NX, N, N+2}; //3D space

    /* Offset of the chunk inside the grid */
    const hsize_t chunk_offset[3] = {X0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(chunk_rank, chunk_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, chunk_offset, NULL, chunk_dims, NULL);

    /* Read the data */
    hid_t h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, data);
    if (h_err < 0) {
        printf("Error: reading chunk of hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    return 0;
}

int readField_dg(struct distributed_grid *dg, const char *fname) {

    /* Open the hdf5 file */
    hid_t h_file = openFile_MPI(dg->comm, fname);

    /* Open the Field group */
    hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Get the file dataspace */
    hid_t h_space = H5Dget_space(h_data);

    /* The chunk in question */
    const hsize_t chunk_rank = 3;
    const hsize_t chunk_dims[3] = {dg->NX, dg->N, dg->N+2}; //3D space

    /* Offset of the chunk inside the grid */
    const hsize_t chunk_offset[3] = {dg->X0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(chunk_rank, chunk_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, chunk_offset, NULL, chunk_dims, NULL);

    /* Read the data */
    hid_t h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, dg->box);
    if (h_err < 0) {
        printf("Error: reading chunk of hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* We read a real box, so the distributed grid is in configuration space */
    dg->momentum_space = 0;

    return 0;
}
