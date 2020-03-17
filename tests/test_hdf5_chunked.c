#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <hdf5.h>

#include "../include/dexm.h"

#define NUMBER_ONE 42
#define NUMBER_TWO 12345
#define M 10
#define N 100000

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
}

void write_test_file(const char *h5fname, const char *simname) {
    /* Create HDF5 file */
    hid_t h_file = H5Fcreate(h5fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    assert(h_file >= 0); //created successfully?

    /* Write another group with a large dataset */
    hid_t h_grp = H5Gcreate(h_file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(h_grp >= 0);

    /* The dataspace consists of MxN random numbers with increasing variance */
    int ndims = 2;
    hsize_t initial_dims[2] = {M/2,N};
    hsize_t max_dims[2] = {H5S_UNLIMITED, N};
    hid_t h_space = H5Screate_simple(ndims, initial_dims, max_dims);
    assert(h_space >= 0);

    /* Dataset properties */
    hid_t h_prop = H5Pcreate(H5P_DATASET_CREATE);
    assert(h_prop >= 0);

    /* Enable a chunked dataset */
    hid_t h_err = H5Pset_layout(h_prop, H5D_CHUNKED);
    assert(h_err >= 0);

    /* Set the size of the first chunk */
    hsize_t chunk_dims[2] = {M/2, N};
    h_err = H5Pset_chunk(h_prop, ndims, chunk_dims);
    assert(h_err >= 0);

    /* Create dataset */
    hid_t h_data = H5Dcreate(h_grp, "Samples", H5T_NATIVE_DOUBLE, h_space, H5P_DEFAULT, h_prop, H5P_DEFAULT);
    assert(h_data >= 0);

    /* Generate the data */
    double *samples = malloc(M/2*N*sizeof(double));
    for (int i=0; i<M/2; i++) {
        for (int j=0; j<N; j++) {
            samples[j + N*i] = sampleNorm() * i;
        }
    }

    /* Create a memory dataspace of the chunk*/
    hid_t h_memspace = H5Screate_simple(ndims, chunk_dims, NULL);
    assert(h_memspace >= 0);

    hsize_t start[2] = {0, 0};
    h_err = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start, NULL, chunk_dims, NULL);
    assert(h_err >= 0);

    /* Write data to the dataset */
    h_err = H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, samples);
    assert(h_err >= 0);

    /* Free the memory */
    free(samples);

    /* Close the dataset and group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);
    H5Pclose(h_prop);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);
}

void append_test_file(const char *h5fname, const char *simname) {
    /* Open the test file */
    hid_t h_file = H5Fopen(h5fname, H5F_ACC_RDWR, H5P_DEFAULT);
    assert(h_file >= 0);

    /* Open the Data group */
    hid_t h_grp = H5Gopen(h_file, "/Data", H5P_DEFAULT);
    assert(h_grp >= 0);

    /* Open the dataset */
    hid_t h_data = H5Dopen(h_grp, "Samples", H5P_DEFAULT);
    assert(h_data >= 0);

    /* Set the size of the second chunk */
    hsize_t chunk_dims[2] = {M/2, N};

    /* The desired total dimensions */
    int ndims = 2;
    hsize_t dims[2] = {M,N};
    hid_t h_err = H5Dset_extent(h_data, dims);
    assert(h_err >= 0);

    /* Generate more data */
    double *samples = malloc(M/2*N*sizeof(double));
    for (int i=0; i<M/2; i++) {
        for (int j=0; j<N; j++) {
            samples[j + N*i] = sampleNorm() * (M/2 + i);
        }
    }

    /* Create a memory dataspace of the chunk*/
    hid_t h_memspace = H5Screate_simple(ndims, chunk_dims, NULL);
    assert(h_memspace >= 0);

    /* Get the file dataspace */
    hid_t h_space = H5Dget_space(h_data);
    assert(h_space >= 0);

    hsize_t start[2] = {M/2, 0};
    h_err = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start, NULL, chunk_dims, NULL);
    assert(h_err >= 0);

    /* Write data to the dataset */
    h_err = H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, samples);
    assert(h_err >= 0);

    /* Close the dataset */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Sclose(h_memspace);

    /* Free the memory */
    free(samples);

    /* Close the Data group */
    H5Gclose(h_grp);

    /* Close the test file */
    H5Fclose(h_file);
}

void read_test_file(const char *h5fname, const char *simname) {
    /* Open the test file */
    hid_t h_file = H5Fopen(h5fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(h_file >= 0);

    /* Open the Data group */
    hid_t h_grp = H5Gopen(h_file, "/Data", H5P_DEFAULT);
    assert(h_grp >= 0);

    /* Open the dataset */
    hid_t h_data = H5Dopen(h_grp, "Samples", H5P_DEFAULT);
    assert(h_data >= 0);

    /* Get the dataspace */
    hid_t h_space = H5Dget_space(h_data);
    assert(h_space >= 0);

    /* We expect two dimensions */
    int ndims = H5Sget_simple_extent_ndims(h_space);
    assert(ndims == 2);

    /* Read out the dimensions */
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* Is it what we expect? */
    assert(dims[0] == M);
    assert(dims[1] == N);

    /* Read out the data */
    double *samples = malloc(M*N*sizeof(double));
    hid_t h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, samples);
    assert(h_err >= 0);

    /* Determine the sample standard deviation of each row */
    printf("sdev: ");
    for (int i=0; i<M; i++) {
        double sum = 0.f;
        for (int j=0; j<N; j++) {
            sum += samples[i*N + j];
        }

        double mean = sum / N;
        double ssum = 0.f;
        for (int j=0; j<N; j++) {
            ssum += (samples[i*N + j] - mean)*(samples[i*N + j] - mean);
        }

        double sdev = sqrt(ssum/(N - 1.0));
        printf("%.2f ", sdev);
        assert(fabs(sdev - i) < 1e-1);
    }
    printf("\n");

    /* Free the memory again */
    free(samples);
    free(dims);

    /* Close the dataset */
    H5Dclose(h_data);
    H5Sclose(h_space);

    /* Close the Data group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);
}

int main() {
    /* Read parameters */
    const char fname[] = "test_cosmology.ini";
    struct params pars;
    struct units us;

    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Seed the random number generator */
    srand(pars.Seed);

    /* File name for the test HDF5 file */
    char h5fname[50];
    sprintf(h5fname, "%s/%s", pars.OutputDirectory, "test_chunked.hdf5");

    /* Test writing an HDF5 file */
    printf("Creating HDF5 file %s\n", h5fname);
    write_test_file(h5fname, pars.Name);

    /* Append some data to the HDF5 file */
    printf("Appending to HDF5 file %s\n", h5fname);
    append_test_file(h5fname, pars.Name);

    /* Test reading an HDF5 file */
    printf("Reading out the HDF5 file %s\n", h5fname);
    read_test_file(h5fname, pars.Name);

    sucmsg("test_hdf5:\t SUCCESS");
}
