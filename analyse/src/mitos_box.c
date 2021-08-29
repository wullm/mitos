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
    message(rank, "Make sure that GridSize and BoxLen are correclty specified.\n");

    /* The size of the density grid that we will create */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Allocate distributed memory arrays (one complex & one real) */
    struct distributed_grid box;
    alloc_local_grid(&box, N, boxlen, MPI_COMM_WORLD);

    /* Read the box */
    int err = readFieldFile_dg(&box, pars.InputFilename);
    if (err > 0) {
        printf("Error reading file '%s'\n.", pars.InputFilename);
        exit(1);
    }

    /* Fourier transform it */
    fft_r2c_dg(&box);


    if (rank == 0) {
        int bins = pars.PowerSpectrumBins;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));

        /* Transform to momentum space */
        fftw_complex *fbox = box.fbox;

        /* Undo the TSC window function */
        // undoTSCWindow(fbox, N, boxlen);

        calc_cross_powerspec(N, boxlen, fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

        /* Check that it is right */
        printf("\n");
        printf("Example power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];


            // double k_long = 0.05;
            // double Bk = calc_cross_bispec(N, boxlen[0], fbox, fbox, fbox, k, k, k, k*0.25, k*0.25, k*0.25);
            // double norm = calc_norm_bispec(N, boxlen[0], k, k, k, k*0.25, k*0.25, k*0.25);

            printf("%f %e %d\n", k, Pk, obs);
        }

        printf("\n");
    }

    /* Free the grids */
    free_local_grid(&box);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
