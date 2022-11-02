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

static inline long long int row_major_padded2(long long i, long long j, long long k, long long N) {
    i = wrap(i,N);
    j = wrap(j,N);
    k = wrap(k,N);
    return (long long int) i*N*(N+2) + j*(N+2) + k;
}


int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftwf_mpi_init();

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
    message(rank, "Reading particle type '%s'.\n", pars.ImportName);
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

    /* Extract the cell structure */
    hid_t h_grp1, h_grp2, h_grp3;
    h_grp1 = H5Gopen(h_file, "Cells", H5P_DEFAULT);
    h_grp2 = H5Gopen(h_grp1, "Meta-data", H5P_DEFAULT);

    /* Read the cell dimensions */
    double cell_dim[3];
    h_attr = H5Aopen(h_grp2, "size", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, cell_dim);
    assert(h_err >= 0);
    H5Aclose(h_attr);

    /* Read the number of cells */
    int cell_num;
    h_attr = H5Aopen(h_grp2, "nr_cells", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_INT, &cell_num);
    assert(h_err >= 0);
    H5Aclose(h_attr);

    /* Read out the cell centres */
    double cell_centres[cell_num][3];
    hid_t h_cdat = H5Dopen(h_grp1, "Centres", H5P_DEFAULT);
    hid_t cstatus = H5Dread(h_cdat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, cell_centres);
    H5Dclose(h_cdat);
    assert(cstatus >= 0);

    /* Read out the particle offsets for the chosen particle type */
    hsize_t cell_p_offsets[cell_num];
    h_grp3 = H5Gopen(h_grp1, "OffsetsInFile", H5P_DEFAULT);
    h_cdat = H5Dopen(h_grp3, pars.ImportName, H5P_DEFAULT);
    cstatus = H5Dread(h_cdat, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, cell_p_offsets);
    H5Dclose(h_cdat);
    H5Gclose(h_grp3);
    assert(cstatus >= 0);

    /* Read out the particle counts for the chosen particle type */
    hsize_t cell_p_counts[cell_num];
    h_grp3 = H5Gopen(h_grp1, "Counts", H5P_DEFAULT);
    h_cdat = H5Dopen(h_grp3, pars.ImportName, H5P_DEFAULT);
    cstatus = H5Dread(h_cdat, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, cell_p_counts);
    H5Dclose(h_cdat);
    H5Gclose(h_grp3);
    assert(cstatus >= 0);

    /* Close the cell groups */
    H5Gclose(h_grp2);
    H5Gclose(h_grp1);

    message(rank, "Cell dimensions are (%g %g %g)\n", cell_dim[0], cell_dim[1], cell_dim[2]);
    message(rank, "There are %d cells.\n", cell_num);
    message(rank, "Cell[0] = %g %g %g with particle offset %lld and count %lld\n", cell_centres[0][0], cell_centres[0][1], cell_centres[0][2], cell_p_offsets[0], cell_p_counts[0]);
    message(rank, "Cell[1] = %g %g %g with particle offset %lld and count %lld\n", cell_centres[1][0], cell_centres[1][1], cell_centres[1][2], cell_p_offsets[1], cell_p_counts[1]);

    /* The size of the density grid that we will create */
    const long int N = pars.GridSize;

    /* Check what portions of 3D grids get stored locally */
    long int X0, NX;
    long int local_size = fftwf_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &NX, &X0);

    /* Allocate the local portion of the grid */
    float *box = fftwf_alloc_real(2*local_size);
    bzero(box, 2*local_size*sizeof(float));

    /* The physical dimensions of the total grid on this rank */
    const double x0 = (boxlen[0] / ((double) N)) * (X0);
    const double x1 = (boxlen[0] / ((double) N)) * (X0 + NX);
    const double delta_x = x1 - x0;


    const double margin = ceil(delta_x * 0.1);

    message(rank, "The safety margin is %g U_L.\n", margin);

    /* Before the main loop, count the number of cells that will be read out */
    int cell_num_local = 0;
    for (int k = 0; k < cell_num; k++) {
        /* The dimensions of the cell */
        const double cell_x0 = cell_centres[k][0] - cell_dim[0] * 0.5;
        const double cell_x1 = cell_x0 + cell_dim[0];

        /* Does this cell contain any particles that belong on this rank? */
        int overlap_x = (x0 - margin <= cell_x0) ? (x1 + margin >= cell_x0) : (x0 - margin <= cell_x1);

        /* Account for wrapping */
        if (x0 <= margin) overlap_x += (cell_x1 >= boxlen[0] - margin);
        if (x1 >= boxlen[0] - margin) overlap_x += (cell_x0 <= margin);

        if (overlap_x)
            cell_num_local++;
    }




    /* Open the corresponding group */
    h_grp = H5Gopen(h_file, pars.ImportName, H5P_DEFAULT);

    //Should we read the masses or assume that all particles have the same mass?
    const int read_masses = 0;

    const double grid_cell_vol = boxlen[0]*boxlen[1]*boxlen[2] / ((long long)N*N*N);
    const double inv_grid_cell_vol = 1. / grid_cell_vol;

    /* Loop over cells */
    int cell_read_local = 0;
    for (int k = 0; k < cell_num; k++) {
        /* The particles in this cell are between offset and offset + count */
        const hsize_t offset = cell_p_offsets[k];
        const hsize_t count = cell_p_counts[k];

        /* Define the hyperslab */
        hsize_t slab_dims[2], start[2]; //for 3-vectors
        hsize_t slab_dims_one[1], start_one[1]; //for scalars

        /* Slab dimensions for 3-vectors */
        slab_dims[0] = count;
        slab_dims[1] = 3; //(x,y,z)
        start[0] = offset;
        start[1] = 0; //start with x

        /* Slab dimensions for scalars */
        slab_dims_one[0] = count;
        start_one[0] = offset;

        /* The dimensions of the cell */
        const double cell_x0 = cell_centres[k][0] - cell_dim[0] * 0.5;
        const double cell_x1 = cell_x0 + cell_dim[0];

        /* Does this cell contain any particles that belong on this rank? */
        int overlap_x = (x0 - margin <= cell_x0) ? (x1 + margin >= cell_x0) : (x0 - margin <= cell_x1);

        /* Account for wrapping */
        if (x0 <= margin) overlap_x += (cell_x1 >= boxlen[0] - margin);
        if (x1 >= boxlen[0] - margin) overlap_x += (cell_x0 <= margin);

        // if (overlap_x && overlap_y && overlap_z)
        if (!overlap_x) continue;

        /* Open the coordinates dataset */
        hid_t h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        hid_t h_space = H5Dget_space(h_dat);

        /* Select the hyperslab */
        hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                           NULL, slab_dims, NULL);
        assert(status >= 0);

        /* Create a memory space */
        hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

        /* Create the data array */
        double data[count][3];

        status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                         data);

        /* Close the memory space */
        H5Sclose(h_mems);

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        /* Create the data array */
        double mass_data[count];

        if (read_masses) {
            /* Open the masses dataset */
            h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);

            /* Find the dataspace (in the file) */
            h_space = H5Dget_space(h_dat);

            /* Select the hyperslab */
            status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                                slab_dims_one, NULL);

            /* Create a memory space */
            h_mems = H5Screate_simple(1, slab_dims_one, NULL);

            status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                             mass_data);

            /* Close the memory space */
            H5Sclose(h_mems);

            /* Close the data and memory spaces */
            H5Sclose(h_space);

            /* Close the dataset */
            H5Dclose(h_dat);
        }

        /* Assign the particles to the grid with CIC */
        const double grid_fac = ((double) N) / boxlen[0];
        for (int l=0; l<count; l++) {
            const double X = data[l][0] * grid_fac;
            const double Y = data[l][1] * grid_fac;
            const double Z = data[l][2] * grid_fac;

            const double m = read_masses ? mass_data[l] : 1.0;

            const long long int iX = (long long int) floor(X);
            const long long int iY = (long long int) floor(Y);
            const long long int iZ = (long long int) floor(Z);

            /* Displacements from grid corner */
            double dx = X - iX;
            double dy = Y - iY;
            double dz = Z - iZ;
            double tx = 1.0 - dx;
            double ty = 1.0 - dy;
            double tz = 1.0 - dz;

            double val = m * inv_grid_cell_vol;

            long long store_x = wrap_ll(iX,N) - X0;
            if (!(store_x < 0 || store_x >= NX)) {
                box[row_major_padded2(store_x, iY, iZ, N)] += val * tx * ty * tz;
                box[row_major_padded2(store_x, iY+1, iZ, N)] += val * tx * dy * tz;
                box[row_major_padded2(store_x, iY, iZ+1, N)] += val * tx * ty * dz;
                box[row_major_padded2(store_x, iY+1, iZ+1, N)] += val * tx * dy * dz;
            }
            long long store_x_p = wrap_ll(iX+1,N) - X0;
            if (!(store_x_p < 0 || store_x_p >= NX)) {
                box[row_major_padded2(store_x_p, iY, iZ, N)] += val * dx * ty * tz;
                box[row_major_padded2(store_x_p, iY+1, iZ, N)] += val * dx * dy * tz;
                box[row_major_padded2(store_x_p, iY, iZ+1, N)] += val * dx * ty * dz;
                box[row_major_padded2(store_x_p, iY+1, iZ+1, N)] += val * dx * dy * dz;
            }


        }

        printf("%3d %.4f%% Read %lld particles from cell %d\n", rank, (100.*cell_read_local/cell_num_local), count, k);
        cell_read_local++;
    }

    double total_mass_local = 0;
    for (long int i = 0; i < 2 * local_size; i++) {
        total_mass_local += box[i];
    }

    total_mass_local /= inv_grid_cell_vol;

    /* Reduce the total mass */
    double total_mass = 0.;
    MPI_Allreduce(&total_mass_local, &total_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    /* The average density */
    double avg_density = total_mass / (boxlen[0]*boxlen[1]*boxlen[2]);

    message(rank, "Total mass: %.10g\n", total_mass);
    message(rank, "Average density %g\n", avg_density);

    /* Turn the density field into an overdensity field */
    for (long int i = 0; i < 2 * local_size; i++) {
        box[i] = (box[i] - avg_density)/avg_density;
    }

    /* Close the group again */
    H5Gclose(h_grp);

    /* Allocate the local portion of the complex grid */
    fftwf_complex *fbox = fftwf_alloc_complex(local_size);

    /* Carry out the FFT with MPI */
    fftwf_plan r2c_mpi = fftwf_mpi_plan_dft_r2c_3d(N, N, N, box, fbox, MPI_COMM_WORLD, FFTW_ESTIMATE);
    fftwf_execute(r2c_mpi);
    fft_normalize_r2c_float_mpi(fbox, N, local_size, boxlen[0]);
    fftwf_destroy_plan(r2c_mpi);


    /* Package the kernel parameter */
    struct Hermite_kern_params Hkp;
    Hkp.order = 2; //CIC
    Hkp.N = N;
    Hkp.boxlen = boxlen[0];

    /* Undo the CIC window function */
    fft_apply_kernel_float_mpi(fbox, fbox, N, X0, NX, boxlen[0], kernel_undo_Hermite_window, &Hkp);
    // undoTSCWindowFloat(fbox, N, boxlen[0]);

    /* Prepare computing the power spectrum with MPI */
    int bins = pars.PowerSpectrumBins;
    double *k_in_bins = malloc(bins * sizeof(double));
    double *power_in_bins = malloc(bins * sizeof(double));
    long int *obs_in_bins = calloc(bins, sizeof(long int));

    /* Compute the power spectrum */
    calc_cross_powerspec_float_mpi(N, X0, NX, boxlen[0], fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

    if (rank == 0) {
        /* Print the result */
        printf("\n");
        printf("Example power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            long int obs = obs_in_bins[i];

            printf("%f %e %ld\n", k, Pk, obs);
        }
        printf("\n");
    }

    /* Free the grids */
    fftwf_free(box);
    fftwf_free(fbox);

    /* Close the HDF5 file */
    H5Fclose(h_file);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
