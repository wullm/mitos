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
#include <math.h>
#include <string.h>
#include "../include/output.h"
#include "../include/output_mpi.h"
#include "../include/fft.h"


hid_t openFile_MPI(MPI_Comm comm, const char *fname) {
    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, comm, MPI_INFO_NULL);

    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDWR, prop_faxs);
    H5Pclose(prop_faxs);

    return h_file;
}

hid_t createFile_MPI(MPI_Comm comm, const char *fname) {
    /* Property list for MPI file access */
    hid_t prop_faxs = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(prop_faxs, comm, MPI_INFO_NULL);

    /* Create the hdf5 file */
    hid_t h_file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, prop_faxs);
    H5Pclose(prop_faxs);

    return h_file;
}

int createFieldGroup_dg(int N, int NX, hid_t h_file) {
    if (NX * N * (N+2) * sizeof(double) > HDF5_PARALLEL_IO_MAX_BYTES) {
        printf("Error: parallel HDF5 cannot handle more than 2GB per chunk.\n");
        return 1;
    }

    /* Create the Field group */
    hid_t h_grp = H5Gcreate(h_file, "/Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for the field */
    const hsize_t frank = 3;
    const hsize_t fdims[3] = {N, N, N+2}; //3D space
    hid_t h_fspace = H5Screate_simple(frank, fdims, NULL);

    /* Create the dataset for the field */
    hid_t h_data = H5Dcreate(h_grp, "Field", H5T_NATIVE_DOUBLE, h_fspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Close the dataset, corresponding dataspace, property list, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_fspace);
    H5Gclose(h_grp);

    return 0;
}

int writeFieldFile_dg(struct distributed_grid *dg, const char *fname) {

        if (dg->momentum_space == 1) {
            printf("Error: attempting to export while in momentum space.\n");
            return 1;
        }

        /* Create the file */
        hid_t h_file = createFile_MPI(dg->comm, fname);

        /* Write the header */
        int err = writeFieldHeader(dg->boxlen, h_file);
        if (err > 0) return err;

        /* Create the Field group */
        err = createFieldGroup_dg(dg->N, dg->NX, h_file);
        if (err > 0) return err;

        /* Write the data */
        err = writeFieldData_dg(dg, h_file);
        if (err > 0) return 0;

        /* Close the file */
        H5Fclose(h_file);

        return 0;
}

int writeFieldData_dg(struct distributed_grid *dg, hid_t h_file) {

    /* Open the Header group */
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

    /* Write the data */
    hid_t h_err = H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, dg->box);
    if (h_err < 0) {
        printf("Error: writing chunk of hdf5 data.\n");
        return 1;
    }

    /* Close the dataset, corresponding dataspace, and the Field group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);
    H5Gclose(h_grp);

    return 0;
}
