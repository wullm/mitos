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
#include <math.h>
#include <hdf5.h>
#include <fftw3.h>
#include <sys/time.h>

#include "../include/dexm.h"

#define printheader(s) printf("\n%s%s%s\n", TXT_BLUE, s, TXT_RESET);

const char *fname;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printheader("DEXM Initial Condition Generator");
    printf("The parameter file is '%s'\n", fname);

    /* Timer */
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    /* DEXM structures */
    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;
    struct perturb_data ptdat;
    struct perturb_spline spline;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, fname);

    printf("The output directory is '%s'.\n", pars.OutputDirectory);
    printf("Creating initial conditions for '%s'.\n", pars.Name);

    /* Read out particle types from the parameter file */
    readTypes(&pars, &types, fname);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat);

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Seed the random number generator */
    srand(pars.Seed);

    /* Determine the starting conformal time */
    cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);

    printheader("Settings");
    printf("Random numbers\t\t [seed] = [%ld]\n", pars.Seed);
    printf("Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
    printf("Primordial power\t [A_s, n_s, k_pivot] = [%.4e, %.4f, %.4f U_L]\n", cosmo.A_s, cosmo.n_s, cosmo.k_pivot);
    printf("\n");

    printf("%s%s%s\n", TXT_BLUE, "Requested Particle Types", TXT_RESET);
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;
        printf("Particle type '%s' (N^3 = %d^3).\n", ptype->Identifier, ptype->CubeRootNumber);
    }


    /* Create Gaussian random field */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Allocate 3D array */
    fftw_complex *grf = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Generate a complex Hermitian Gaussian random field */
    printheader("Generating Primordial Fluctuations");
    generate_complex_grf(grf, N, boxlen);

    /* Apply the bare power spectrum, without any transfer functions */
    fft_apply_kernel(grf, grf, N, boxlen, kernel_power_no_transfer, &cosmo);

    /* Convert from complex phases to real Gaussian variates and export the box */
    char box_fname[DEFAULT_STRING_LENGTH];
    sprintf(box_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");
    fft_c2r_export(grf, N, boxlen, box_fname);
    printf("Pure Gaussian Random Field exported to '%s'.\n", box_fname);


    /* For each particle type, fetch the user-defined density function title */
    printheader("Fetching Density Perturbations");
    char **function_titles = malloc(pars.NumParticleTypes * sizeof(char*));
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;
        function_titles[pti] = ptype->TransferFunctionDensity;
        printf("Particle type '%s' uses density vector '%s'.\n", Identifier, function_titles[pti]);
    }

    /* Generate the density grids */
    printheader("Generating Density Grids");
    int err = generatePerturbationGrids(&pars, &us, &cosmo, &spline, types, grf, function_titles, GRID_NAME_DENSITY);
    if (err > 0) exit(1);

    /* For each particle type, fetch the user-defined energy flux function title */
    printheader("Fetching Energy Flux Perturbations");
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;
        function_titles[pti] = ptype->TransferFunctionVelocity;
        printf("Particle type '%s' uses energy flux vector '%s'.\n", Identifier, function_titles[pti]);
    }

    /* Generate the energy flux (velocity divergence theta) grids */
    printheader("Generating Energy Flux Fields");
    err = generatePerturbationGrids(&pars, &us, &cosmo, &spline, types, grf, function_titles, GRID_NAME_THETA);
    if (err > 0) exit(1);

    /* Get rid of the perturbation vector function titles */
    free(function_titles);

    /* Get rid of the random phases field */
    fftw_free(grf);

    /* Compute the potential grids */
    printheader("Computing Gravitational Potentials");
    err = computePotentialGrids(&pars, &us, &cosmo, types);
    if (err > 0) exit(1);

    /* Compute derivatives of the potential grids */
    printheader("Computing Potential Derivatives (Displacements)");
    err = computeGridDerivatives(&pars, &us, &cosmo, types, GRID_NAME_DENSITY, GRID_NAME_DISPLACEMENT);
    if (err > 0) exit(1);

    /* Compute derivatives of the energy flux grids */
    printheader("Computing Energy Flux Derivatives (Velocities)");
    err = computeGridDerivatives(&pars, &us, &cosmo, types, GRID_NAME_THETA, GRID_NAME_VELOCITY);
    if (err > 0) exit(1);

    /* Name of the main output file containing the initial conditions */
    printheader("Initializing Output File");
    char out_fname[DEFAULT_STRING_LENGTH];
    sprintf(out_fname, "%s/%s", pars.OutputDirectory, pars.OutputFilename);
    printf("Creating output file '%s'.\n", out_fname);

    /* Create the output file */
    hid_t h_out_file = H5Fcreate(out_fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_out_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);
    H5Aclose(h_attr);
    H5Sclose(h_aspace);


    /* Create dataspace for BoxSize attribute */
    const hsize_t adims2[1] = {6}; //6 particle types
    h_aspace = H5Screate_simple(arank, adims2, NULL);

    /* Create the BoxSize attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    long long int numparts[6] = {0, 0, 0, 0, 0, 0};
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts);
    H5Aclose(h_attr);
    H5Sclose(h_aspace);

    /* Close the Header group */
    H5Gclose(h_grp);

    /* Counter of total number of particle stored */
    long long int grand_counter = 0;

    /* For each user-defined particle type */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;

        printf("\n%s%s%s%s%s\n", TXT_BLUE, "Generating Particle Type '", ptype->Identifier, "'.", TXT_RESET);

        /* Skip empty particle types */
        if (ptype->TotalNumber <= 0) {
            printf("No particles requested.\n");
            continue;
        }

        /* ID of the first particle of this type */
        const long long int id_first_particle = grand_counter;
        grand_counter += ptype->TotalNumber;

        /* Create the particle group in the output file */
        char *gname = ptype->ExportName;
        hid_t h_grp = H5Gcreate(h_out_file, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        /* Vector dataspace (e.g. positions, velocities) */
        const hsize_t vrank = 2;
        const hsize_t vdims[2] = {ptype->TotalNumber, 3};
        hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

        /* Scalar dataspace (e.g. masses, particle ids) */
        const hsize_t srank = 1;
        const hsize_t sdims[1] = {ptype->TotalNumber};
        hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

        /* Create various datasets (empty for now) */
        hid_t h_data;

        /* Coordinates (use vector space) */
        h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Velocities (use vector space) */
        h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Masses (use scalar space) */
        h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Particle IDs (use scalar space) */
        h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(h_data);

        /* Allocate enough memory for one chunk of particles */
        struct particle *parts;
        allocParticles(&parts, &pars, ptype);

        /* Allocate memory for the displacement and velocity grids */
        double *grid = malloc(N*N*N * sizeof(double));

        /* For each chunk, generate and store the particles */
        for (int chunk=0; chunk<ptype->Chunks; chunk++) {
            /* The dimensions of this chunk */
            const hsize_t start = chunk * ptype->ChunkSize;
            const hsize_t remaining = ptype->TotalNumber - start;
            const hsize_t chunk_size = (hsize_t) fmin(ptype->ChunkSize, remaining);

            printf("Generating chunk %d.\n", chunk);
            genParticles_FromGrid(&parts, &pars, &us, &cosmo, ptype, chunk, id_first_particle);

            /* Interpolating displacements at the pre-initial particle locations */
            /* For x, y, and z */
            const char letters[] = {'x', 'y', 'z'};
            for (int dir=0; dir<3; dir++) {
                char dbox_fname[DEFAULT_STRING_LENGTH];
                sprintf(dbox_fname, "%s/%s_%c_%s%s", pars.OutputDirectory, GRID_NAME_DISPLACEMENT, letters[dir], ptype->Identifier, ".hdf5");
                printf("Displacement field read from '%s'.\n", dbox_fname);
                int err = readGRF_inPlace_H5(grid, dbox_fname);
                if (err > 0) exit(1);

                /* Displace the particles in this chunk */
                #pragma omp parallel for
                for (int i=0; i<chunk_size; i++) {
                    /* Find the pre-initial (e.g. grid) locations */
                    double x = parts[i].X;
                    double y = parts[i].Y;
                    double z = parts[i].Z;

                    /* Find the displacement */
                    double disp = gridTSC(grid, N, boxlen, x, y, z);

                    /* Displace the particles */
                    if (dir == 0) {
                        parts[i].X -= disp;
                    } else if (dir == 1) {
                        parts[i].Y -= disp;
                    } else {
                        parts[i].Z -= disp;
                    }
                }
            }

            /* Interpolating velocities at the displaced particle locations */
            /* For x, y, and z */
            for (int dir=0; dir<3; dir++) {
                char dbox_fname[DEFAULT_STRING_LENGTH];
                sprintf(dbox_fname, "%s/%s_%c_%s%s", pars.OutputDirectory, GRID_NAME_VELOCITY, letters[dir], ptype->Identifier, ".hdf5");
                printf("Velocity field read from '%s'.\n", dbox_fname);
                int err = readGRF_inPlace_H5(grid, dbox_fname);
                if (err > 0) exit(1);

                /* Displace the particles in this chunk */
                #pragma omp parallel for
                for (int i=0; i<chunk_size; i++) {
                    /* Find the displaceed particle location */
                    double x = parts[i].X;
                    double y = parts[i].Y;
                    double z = parts[i].Z;

                    /* Find the velocity in the given direction */
                    double vel = gridTSC(grid, N, boxlen, x, y, z);

                    /* Displace the particles */
                    if (dir == 0) {
                        parts[i].v_X = vel;
                    } else if (dir == 1) {
                        parts[i].v_Y = vel;
                    } else {
                        parts[i].v_Z = vel;
                    }
                }
            }


            /* Unit conversions */
            /* (...) */

            /* Before writing particle data to disk, we need to choose the
             * hyperslabs, i.e. the parts of memory where the data is stored.
             * In our case, these correspond to the contiguous chunks of particle
             * data, specified by a start and a dimensions vector.
             */

            /* Create scalar & vector datapsace for smaller chunks of data */
            const hsize_t ch_vdims[2] = {chunk_size, 3};
            const hsize_t ch_sdims[2] = {chunk_size};
            hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
            hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

            /* The start of this chunk, in the overall vector & scalar spaces */
            const hsize_t vstart[2] = {start, 0}; //always with the "x" coordinate
            const hsize_t sstart[1] = {start};

            /* Choose the corresponding hyperslabs inside the overall spaces */
            H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
            H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

            /* Unpack particle data into contiguous arrays */
            double *coords = malloc(3 * chunk_size * sizeof(double));
            double *vels = malloc(3 * chunk_size * sizeof(double));
            double *masses = malloc(1 * chunk_size * sizeof(double));
            long long *ids = malloc(1 * chunk_size * sizeof(long long));
            for (int i=0; i<chunk_size; i++) {
                coords[i * 3 + 0] = parts[i].X;
                coords[i * 3 + 1] = parts[i].Y;
                coords[i * 3 + 2] = parts[i].Z;
                vels[i * 3 + 0] = parts[i].v_X;
                vels[i * 3 + 1] = parts[i].v_Y;
                vels[i * 3 + 2] = parts[i].v_Z;
                masses[i] = parts[i].mass;
                ids[i] = parts[i].id;
            }

            /* Write coordinate data (vector) */
            h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
            H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
            H5Dclose(h_data);
            free(coords);

            /* Write velocity data (vector) */
            h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
            H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
            H5Dclose(h_data);
            free(vels);

            /* Write mass data (scalar) */
            h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
            H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
            H5Dclose(h_data);
            free(masses);

            /* Write particle id data (scalar) */
            h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
            H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
            H5Dclose(h_data);
            free(ids);

            /* Close the chunk-sized scalar and vector dataspaces */
            H5Sclose(h_ch_vspace);
            H5Sclose(h_ch_sspace);

        }

        /* Free memory of the displacement and velocity grids */
        free(grid);

        /* Close the scalar and vector dataspaces */
        H5Sclose(h_vspace);
        H5Sclose(h_sspace);

        /* Clean the particles up */
        cleanParticles(&parts, &pars, ptype);

        /* Close the group in the output file */
        H5Gclose(h_grp);
    }

    /* Close the output file */
    H5Fclose(h_out_file);


    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
    cleanPerturb(&ptdat);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);

    /* Timer */
    gettimeofday(&stop, NULL);
    long unsigned microsec = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\nTime elapsed: %.3f ms\n", microsec/1000.);

}
