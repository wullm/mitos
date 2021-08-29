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
#include <math.h>
#include "../include/input.h"
#include "../include/fft.h"

int readParams(struct params *pars, const char *fname) {
     pars->Seed = ini_getl("Random", "Seed", 1, fname);

     pars->GridSize = ini_getl("Box", "GridSize", 64, fname);
     pars->SmallGridSize = ini_getl("Box", "SmallGridSize", 0, fname);
     pars->BoxLen = ini_getd("Box", "BoxLen", 1.0, fname);
     pars->Splits = ini_getl("Box", "Splits", 1, fname);
     pars->NeighbourSliverSize = ini_getl("Box", "NeighbourSliverSize", 6, fname);


     pars->MaxParticleTypes = ini_getl("Simulation", "MaxParticleTypes", 1, fname);
     pars->NumParticleTypes = 0; //should not be read, but inferred
     pars->Homogeneous = ini_getbool("Simulation", "Homogeneous", 0, fname);
     pars->MergeDarkMatterBaryons = ini_getbool("PerturbData", "MergeDarkMatterBaryons", 0, fname);
     pars->GrowthFactorsFromSecondFile = ini_getbool("PerturbData", "GrowthFactorsFromSecondFile", 0, fname);
     pars->SlabSize = ini_getl("Read", "SlabSize", 8000000, fname);
     pars->HaloMinMass = ini_getd("Read", "HaloMinMass", 2.75e4, fname);
     pars->HaloMaxMass = ini_getd("Read", "HaloMaxMass", 2.75e5, fname);
     pars->PowerSpectrumBins = ini_getl("Read", "PowerSpectrumBins", 50, fname);
     pars->LossyScaleDigits = ini_getl("Output", "LossyScaleDigits", 0, fname);
     pars->MitosDerivScaling = ini_getd("Read", "MitosDerivScaling", 1e-7, fname);

     /* Bispectrum parameters (default = equilateral mode) */
     pars->BispectrumType = ini_getl("Read", "BispectrumType", 0, fname);
     pars->BispectrumAngle = ini_getd("Read", "BispectrumAngle", M_PI * (4./3.), fname);
     pars->BispectrumMode = ini_getd("Read", "BispectrumAngle", 0.01, fname);
     pars->BispectrumSecondBins = ini_getl("Read", "BispectrumSecondBins", 0, fname);

     /* Read strings */
     int len = DEFAULT_STRING_LENGTH;
     pars->OutputDirectory = malloc(len);
     pars->Name = malloc(len);
     pars->InputFilename = malloc(len);
     pars->InputFilename2 = malloc(len);
     pars->HaloInputFilename = malloc(len);
     pars->ImportName = malloc(len);
     pars->OutputFilename = malloc(len);
     pars->PerturbFile = malloc(len);
     pars->SecondPerturbFile = malloc(len);
     pars->SwiftParamFilename = malloc(len);
     pars->CrossSpectrumDensity1 = malloc(len);
     pars->CrossSpectrumDensity2 = malloc(len);
     pars->ReadGaussianFileName = malloc(len);
     ini_gets("Output", "Directory", "./output", pars->OutputDirectory, len, fname);
     ini_gets("Simulation", "Name", "No Name", pars->Name, len, fname);
     ini_gets("Output", "Filename", "particles.hdf5", pars->OutputFilename, len, fname);
     ini_gets("Output", "SwiftParamFilename", "swift_params.hdf5", pars->SwiftParamFilename, len, fname);
     ini_gets("PerturbData", "File", "", pars->PerturbFile, len, fname);
     ini_gets("PerturbData", "SecondFile", "", pars->SecondPerturbFile, len, fname);
     ini_gets("Read", "Filename", "", pars->InputFilename, len, fname);
     ini_gets("Read", "Filename2", "", pars->InputFilename2, len, fname);
     ini_gets("Read", "ImportName", "", pars->ImportName, len, fname);
     ini_gets("Read", "HaloFilename", "", pars->HaloInputFilename, len, fname);
     ini_gets("Read", "CrossSpectrumDensity1", "", pars->CrossSpectrumDensity1, len, fname);
     ini_gets("Read", "CrossSpectrumDensity2", "", pars->CrossSpectrumDensity2, len, fname);
     ini_gets("Read", "ReadGaussianFileName", "", pars->ReadGaussianFileName, len, fname);

     /* Read optional settings for the Firebolt Boltzmann solver */
     pars->MaxMultipole = ini_getl("Firebolt", "MaxMultipole", 2000, fname);
     pars->MaxMultipoleConvert = ini_getl("Firebolt", "MaxMultipoleConvert", 2, fname);
     pars->NumberMomentumBins = ini_getl("Firebolt", "NumberMomentumBins", 10, fname);
     pars->NumberWavenumbers = ini_getl("Firebolt", "NumberWavenumbers", 10, fname);
     pars->FireboltCutoffWavenumber = ini_getd("Firebolt", "FireboltCutoffWavenumber", 1, fname);
     pars->MinMomentum = ini_getd("Firebolt", "MinMomentum", 0.01, fname);
     pars->MaxMomentum = ini_getd("Firebolt", "MaxMomentum", 15, fname);
     pars->FireboltTolerance = ini_getd("Firebolt", "Tolerance", 1e-10, fname);
     pars->FireboltVerbose = ini_getl("Firebolt", "Verbose", 0, fname);
     pars->FireboltGridSize = ini_getl("Firebolt", "FireboltGridSize", 0, fname);

     return 0;
}

int readUnits(struct units *us, const char *fname) {
    /* Internal units */
    us->UnitLengthMetres = ini_getd("Units", "UnitLengthMetres", 1.0, fname);
    us->UnitTimeSeconds = ini_getd("Units", "UnitTimeSeconds", 1.0, fname);
    us->UnitMassKilogram = ini_getd("Units", "UnitMassKilogram", 1.0, fname);
    us->UnitTemperatureKelvin = ini_getd("Units", "UnitTemperatureKelvin", 1.0, fname);
    us->UnitCurrentAmpere = ini_getd("Units", "UnitCurrentAmpere", 1.0, fname);

    /* Get the transfer functions format */
    char format[DEFAULT_STRING_LENGTH];
    ini_gets("TransferFunctions", "Format", "Plain", format, DEFAULT_STRING_LENGTH, fname);

    /* Format of the transfer functions */
    int default_h_exponent;
    int default_k_exponent;
    int default_sign;
    if (strcmp(format, "CLASS") == 0) {
        default_h_exponent = 1;
        default_k_exponent = 0;
        default_sign = -1;
    } else {
        default_h_exponent = 0;
        default_k_exponent = -2;
        default_sign = +1;
    }
    us->TransferUnitLengthMetres = ini_getd("TransferFunctions", "UnitLengthMetres", MPC_METRES, fname);
    us->Transfer_hExponent = ini_getl("TransferFunctions", "hExponent", default_h_exponent, fname);
    us->Transfer_kExponent = ini_getl("TransferFunctions", "kExponent", default_k_exponent, fname);
    us->Transfer_Sign = ini_getl("TransferFunctions", "Sign", default_sign, fname);

    /* Some physical constants */
    us->SpeedOfLight = SPEED_OF_LIGHT_METRES_SECONDS * us->UnitTimeSeconds
                        / us->UnitLengthMetres;
    us->GravityG = GRAVITY_G_SI_UNITS * us->UnitTimeSeconds * us->UnitTimeSeconds
                    / us->UnitLengthMetres / us->UnitLengthMetres / us->UnitLengthMetres
                    * us->UnitMassKilogram; // m^3 / kg / s^2 to internal
    us->hPlanck = PLANCK_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds; //J*s = kg*m^2/s
    us->kBoltzmann = BOLTZMANN_CONST_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds * us->UnitTimeSeconds
                    * us->UnitTemperatureKelvin; //J/K = kg*m^2/s^2/K
    us->ElectronVolt = ELECTRONVOLT_SI_UNITS / us->UnitMassKilogram / us->UnitLengthMetres
                    / us->UnitLengthMetres * us->UnitTimeSeconds
                    * us->UnitTimeSeconds; // J = kg*m^2/s^2

    return 0;
}

int readCosmology(struct cosmology *cosmo, struct units *us, const char *fname) {
     cosmo->h = ini_getd("Cosmology", "h", 0.70, fname);
     cosmo->n_s = ini_getd("Cosmology", "n_s", 0.97, fname);
     cosmo->A_s = ini_getd("Cosmology", "A_s", 2.215e-9, fname);
     cosmo->k_pivot = ini_getd("Cosmology", "k_pivot", 0.05, fname);
     cosmo->z_ini = ini_getd("Cosmology", "z_ini", 40.0, fname);

     /* Default value for z_source is z_ini */
     cosmo->z_source = ini_getd("Cosmology", "z_source", cosmo->z_ini, fname);

     double H0 = 100 * cosmo->h * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
     cosmo->rho_crit = 3 * H0 * H0 / (8 * M_PI * us->GravityG);

     return 0;
}

int cleanParams(struct params *pars) {
    free(pars->OutputDirectory);
    free(pars->Name);
    free(pars->InputFilename);
    free(pars->InputFilename2);
    free(pars->HaloInputFilename);
    free(pars->OutputFilename);
    free(pars->PerturbFile);
    free(pars->SecondPerturbFile);
    free(pars->SwiftParamFilename);
    free(pars->CrossSpectrumDensity1);
    free(pars->CrossSpectrumDensity2);
    free(pars->ReadGaussianFileName);

    return 0;
}

/* Read 3D box from disk, allocating memory and storing the grid dimensions */
int readFieldFile(double **box, int *N, double *box_len, const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the size of the field */
    hid_t h_attr, h_err;
    double boxsize[3];

    /* Open and read out the attribute */
    h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
    if (h_err < 0) {
        printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
        return 1;
    }

    /* It should be a cube */
    assert(boxsize[0] == boxsize[1]);
    assert(boxsize[1] == boxsize[2]);
    *box_len = boxsize[0];

    /* Close the attribute, and the Header group */
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Open the Field group */
    h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);
    int read_N = dims[0];

    /* We should be in 3D */
    if (ndims != 3) {
        printf("Number of dimensions %d != 3.\n", ndims);
        return 2;
    }
    /* It should be a cube (but allow for padding in the last dimension) */
    if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
        printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
        return 2;
    }
    /* Store the grid size */
    *N = read_N;

    /* Allocate the array (without padding) */
    *box = malloc(read_N * read_N * read_N * sizeof(double));

    /* The hyperslab that should be read (needed in case of padding) */
    const hsize_t space_rank = 3;
    const hsize_t space_dims[3] = {read_N, read_N, read_N}; //3D space

    /* Offset of the hyperslab */
    const hsize_t space_offset[3] = {0, 0, 0};

    /* Create memory space for the chunk */
    hid_t h_memspace = H5Screate_simple(space_rank, space_dims, NULL);
    H5Sselect_hyperslab(h_space, H5S_SELECT_SET, space_offset, NULL, space_dims, NULL);

    /* Read out the data */
    h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, h_memspace, h_space, H5P_DEFAULT, *box);
    if (h_err < 0) {
        printf("Error reading hdf5 file '%s'.\n", fname);
        return 1;
    }

    /* Close the dataspaces and dataset */
    H5Sclose(h_memspace);
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}

/* Read the box without any checks, assuming we have sufficient memory
 * allocated. */
int readFieldFileInPlace(double *box, const char *fname) {
    /* Create the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Field group */
    hid_t h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);

    /* Read out the data */
    hid_t h_err = H5Dread(h_data, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, box);
    if (h_err < 0) {
        printf("Error reading hdf5 file '%s' in place.\n", fname);
        return 1;
    }

    /* Close the dataspace and dataset */
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}

int readFieldDimensions(int *N, double *box_len, const char *fname) {
    /* Open the hdf5 file */
    hid_t h_file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the size of the field */
    hid_t h_attr, h_err;
    double boxsize[3];

    /* Open and read out the attribute */
    h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &boxsize);
    if (h_err < 0) {
        printf("Error reading hdf5 attribute '%s'.\n", "BoxSize");
        return 1;
    }

    /* It should be a cube */
    assert(boxsize[0] == boxsize[1]);
    assert(boxsize[1] == boxsize[2]);
    *box_len = boxsize[0];

    /* Close the attribute, and the Header group */
    H5Aclose(h_attr);
    H5Gclose(h_grp);

    /* Open the Field group */
    h_grp = H5Gopen(h_file, "Field", H5P_DEFAULT);

    /* Open the Field dataset */
    hid_t h_data = H5Dopen2(h_grp, "Field", H5P_DEFAULT);

    /* Open the dataspace and fetch the grid dimensions */
    hid_t h_space = H5Dget_space(h_data);
    int ndims = H5Sget_simple_extent_ndims(h_space);
    hsize_t *dims = malloc(ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(h_space, dims, NULL);
    int read_N = dims[0];

    /* We should be in 3D */
    if (ndims != 3) {
        printf("Number of dimensions %d != 3.\n", ndims);
        return 2;
    }
    /* It should be a cube (but allow for padding in the last dimension) */
    if (read_N != dims[1] || (read_N != dims[2] && (read_N+2) != dims[2])) {
        printf("Non-cubic grid size (%lld, %lld, %lld).\n", dims[0], dims[1], dims[2]);
        return 2;
    }
    /* Store the grid size */
    *N = read_N;

    /* Close the dataspace and dataset */
    H5Sclose(h_space);
    H5Dclose(h_data);

    /* Close the Field group */
    H5Gclose(h_grp);

    /* Close the file */
    H5Fclose(h_file);

    /* Free memory */
    free(dims);

    return 0;
}
