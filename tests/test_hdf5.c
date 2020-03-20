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

    /* Open new Header group to write simulation properties */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(h_grp >= 0);

    /* Create dataspace */
    hid_t h_space = H5Screate(H5S_SIMPLE);
    assert(h_space >= 0);

    /* Write an attribute consisting of two integers */
    int ndims = 1;
    hsize_t dim[1] = {2};
    hid_t h_err = H5Sset_extent_simple(h_space, ndims, dim, NULL);
    assert(h_err >= 0);

    /* Create the attribute */
    hid_t h_attr = H5Acreate1(h_grp, "Two Numbers", H5T_NATIVE_INT, h_space, H5P_DEFAULT);
    assert(h_attr >= 0);

    /* Write the data to the attribute */
    int data[2] = {NUMBER_ONE,NUMBER_TWO};
    h_err = H5Awrite(h_attr, H5T_NATIVE_INT, data);
    assert(h_err >= 0);

    /* Done with this attribute */
    H5Sclose(h_space);
    H5Aclose(h_attr);

    /* Create another dataspace for a second attribute */
    h_space = H5Screate(H5S_SCALAR);
    assert(h_space >= 0);

    /* For strings, we need to prepare a datatype */
    const hid_t h_type = H5Tcopy(H5T_C_S1);
    assert(h_type >= 0);
    h_err = H5Tset_size(h_type, strlen(simname)); //length of simname
    assert(h_err >= 0);

    /* Create the second attribute, which is the name of the simulation */
    h_attr = H5Acreate1(h_grp, "Name", h_type, h_space, H5P_DEFAULT);
    assert(h_attr >= 0);

    /* Write the second attribute */
    h_err = H5Awrite(h_attr, h_type, simname);
    assert(h_err >= 0);

    /* Done with the Header group */
    H5Tclose(h_type);
    H5Sclose(h_space);
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Write another group with a large dataset */
    h_grp = H5Gcreate(h_file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(h_grp >= 0);

    /* Create another dataspace */
    h_space = H5Screate(H5S_SIMPLE);
    assert(h_space >= 0);

    /* The dataspace consists of MxN random numbers with increasing variance */
    ndims = 2;
    hsize_t dim2[2] = {M,N};
    h_err = H5Sset_extent_simple(h_space, ndims, dim2, NULL);
    assert(h_err >= 0);

    /* Dataset properties */
    hid_t h_prop = H5Pcreate(H5P_DATASET_CREATE);
    assert(h_prop >= 0);

    /* Create dataset */
    hid_t h_data = H5Dcreate(h_grp, "Samples", H5T_NATIVE_DOUBLE, h_space, H5P_DEFAULT, h_prop, H5P_DEFAULT);
    assert(h_data >= 0);

    /* Generate the data */
    double *samples = malloc(M*N*sizeof(double));
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            samples[j + N*i] = sampleNorm() * i;
        }
    }

    /* Write data to the dataset */
    h_err = H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_space, H5S_ALL, H5P_DEFAULT, samples);
    assert(h_err >= 0);

    free(samples);

    /* Close the dataset and group */
    H5Dclose(h_data);
    H5Sclose(h_space);
    H5Pclose(h_prop);
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);
}

void read_test_file(const char *h5fname, const char *simname) {
    /* Open the test file */
    hid_t h_file = H5Fopen(h5fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(h_file >= 0);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "/Header", H5P_DEFAULT);
    assert(h_grp >= 0);

    /* Open the attribute, consisting of two numbers */
    hid_t h_attr = H5Aopen(h_grp, "Two Numbers", H5P_DEFAULT);
    assert(h_attr >= 0);

    /* Read out the values */
    int data[2];
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_INT, data);
    assert(h_err >= 0);

    /* Is it what we expected? */
    assert(data[0] == NUMBER_ONE);
    assert(data[1] == NUMBER_TWO);

    /* Close the attribute again */
    H5Aclose(h_attr);

    /* Open the string attribute */
    h_attr = H5Aopen(h_grp, "Name", H5P_DEFAULT);
    assert(h_attr >= 0);

    /* Get the datatype of the string */
    hid_t h_tp = H5Aget_type(h_attr);
    assert(h_tp >= 0);

    /* Read out the string */
    char str[50];
    h_err = H5Aread(h_attr, h_tp, str);

    printf("The name stored was \"%s\".\n", str);
    assert(strcmp(str, "Test Simulation") == 0);

    /* Close the second attribute */
    H5Tclose(h_tp);
    H5Aclose(h_attr);

    /* Close the header group */
    H5Gclose(h_grp);

    /* Open the Data group */
    h_grp = H5Gopen(h_file, "/Data", H5P_DEFAULT);
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
    h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, samples);

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
    sprintf(h5fname, "%s/%s", pars.OutputDirectory, "test.hdf5");

    /* Test writing an HDF5 file */
    printf("Creating HDF5 file %s\n", h5fname);
    write_test_file(h5fname, pars.Name);

    /* Test reading an HDF5 file */
    printf("Reading out the HDF5 file %s\n", h5fname);
    read_test_file(h5fname, pars.Name);

    /* Clean up */
    cleanParams(&pars);

    sucmsg("test_hdf5:\t SUCCESS");
}
