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

#include "../include/dexm.h"

const char *fname;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printf("The parameter file is '%s'\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;
    struct transfer trs;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, fname);
    readTypes(&pars, &types, fname);
    readTransfers(&pars, &us, &cosmo, &trs);

    /* Initialize the interpolation splines for the transfer functions */
    tr_interp_init(&trs);

    /* Initialize the primordial power spectrum function */
    initPrimordial(&pars, &cosmo);

    printf("Creating initial conditions for '%s'.\n", pars.Name);

    /* Seed the random number generator */
    srand(pars.Seed);


    /* Create Gaussian random field */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Create 3D arrays */
    double *box = (double*) fftw_malloc(N*N*N*sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fboxcpy = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate a complex Hermitian Gaussian random field */
    generate_complex_grf(fbox, N, boxlen);

    /* Copy the complex random field for use later */
    memcpy(fboxcpy, fbox, N*N*(N/2+1)*sizeof(fftw_complex));

    /* Transform to real configuraiton space */
    fft_execute(c2r);
	fft_normalize_c2r(box,N,boxlen);

    /* Export the Gaussian random field for testing purposes */
    char box_fname[DEFAULT_STRING_LENGTH];
    sprintf(box_fname, "%s/%s", pars.OutputDirectory, "gaussian_pure.hdf5");
    write_doubles_as_floats("box.box", box, N*N*N);
    writeGRF_H5(box, N, boxlen, box_fname);
    printf("Pure Gaussian random field exported to '%s'.\n", box_fname);

    printf("\n");

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* The user-defined title of the density transfer function */
        const char *title = ptype->TransferFunctionDensity;

        /* Find the title among the transfer functions */
        int trfunc_id = find_title(trs.titles, title, trs.n_functions);
        if (trfunc_id < 0) {
            printf("Error: transfer function '%s' not found (%d).\n", title, trfunc_id);
            exit(1);
        }

        /* Switch the interpolation spline to this transfer function */
        tr_interp_switch_func(&trs, trfunc_id);

        /* Restore the original pure complex Gaussian random field */
        memcpy(fbox, fboxcpy, N*N*(N/2+1)*sizeof(fftw_complex));

        /* Apply the transfer function to fbox */
        fft_apply_kernel(fbox, fbox, N, boxlen, fullPowerSpectrumKernel);

        /* Transform to real configuration space */
        fft_execute(c2r);
    	fft_normalize_c2r(box,N,boxlen);

        /* Write the field to disk */
        char dbox_fname[DEFAULT_STRING_LENGTH];
        char dbox_fname2[DEFAULT_STRING_LENGTH];
        sprintf(dbox_fname, "%s/%s%s%s", pars.OutputDirectory, "density_", Identifier, ".hdf5");
        sprintf(dbox_fname2, "%s/%s%s%s", pars.OutputDirectory, "density_", Identifier, ".box");
        write_doubles_as_floats(dbox_fname2, box, N*N*N);
        writeGRF_H5(box, N, boxlen, dbox_fname);
        printf("Density field '%s' exported to '%s'.\n", title, dbox_fname);
    }

    /* Free all the FFT objects */
    fftw_free(box);
    fftw_free(fbox);
    fftw_free(fboxcpy);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    printf("\n");

    /* Name of the main output file containing the initial conditions */
    char out_fname[DEFAULT_STRING_LENGTH];
    sprintf(out_fname, "%s/%s", pars.OutputDirectory, pars.OutputFilename);
    printf("Creating output file '%s'.\n", out_fname);

    /* Create the output file */
    hid_t h_out_file = H5Fcreate(out_fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);


    /* For each user-defined particle type */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;

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

        printf("\n");
        printf("Particle type '%s'.\n", ptype->Identifier);

        /* Allocate enough memory for one chunk of particles */
        struct particle *parts;
        allocParticles(&parts, &pars, ptype);

        /* For each chunk, generate and store the particles */
        for (int chunk=0; chunk<ptype->Chunks; chunk++) {
            /* The dimensions of this chunk */
            const hsize_t start = chunk * ptype->ChunkSize;
            const hsize_t remaining = ptype->TotalNumber - start;
            const hsize_t chunk_size = (hsize_t) fmin(ptype->ChunkSize, remaining);

            printf("Generating chunk %d.\n", chunk);
            genParticles_FromGrid(&parts, &pars, &us, &cosmo, ptype, chunk);

            /* Do all imaginable manipulations to these particles */
            /* (...) */

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
    tr_interp_free(&trs);
    cleanTransfers(&trs);
    cleanTypes(&pars, &types);
    cleanParams(&pars);

}
