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
#include <complex.h>

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
    struct perturb_data ptdat;
    struct perturb_params ptpars;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat, pars.PerturbFile);
    readPerturbParams(&pars, &us, &ptpars, pars.PerturbFile);

    /* Merge cdm & baryons into one set of transfer functions (replacing cdm) */
    if (pars.MergeDarkMatterBaryons) {
        header(rank, "Merging cdm & baryon transfer functions, replacing cdm.");

        /* The indices of the density transfer functions */
        int index_cdm = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
        int index_b = findTitle(ptdat.titles, "d_b", ptdat.n_functions);

        /* Find the present-day background densities */
        int today_index = ptdat.tau_size - 1; // today corresponds to the last index
        double Omega_cdm = ptdat.Omega[ptdat.tau_size * index_cdm + today_index];
        double Omega_b = ptdat.Omega[ptdat.tau_size * index_b + today_index];

        /* Do a sanity check */
        assert(fabs(Omega_b - ptpars.Omega_b) / Omega_b < 1e-5);

        /* Use the present-day densities as weights */
        double weight_cdm = Omega_cdm / (Omega_cdm + Omega_b);
        double weight_b = Omega_b / (Omega_cdm + Omega_b);

        message(rank, "Using weights [w_cdm, w_b] = [%f, %f]\n", weight_cdm, weight_b);

        /* Merge the density & velocity transfer runctions, replacing cdm */
        mergeTransferFunctions(&ptdat, "d_cdm", "d_b", weight_cdm, weight_b);
        mergeTransferFunctions(&ptdat, "t_cdm", "t_b", weight_cdm, weight_b);
        /* Merge the background densities, replacing cdm */
        mergeBackgroundDensities(&ptdat, "d_cdm", "d_b", 1.0, 1.0); //replace with sum
    }

    message(rank, "Reading simulation snapshot for: \"%s\".\n", pars.Name);
    message(rank, "Make sure that GridSize and BoxLen are correclty specified.\n");


    /* Open the first mesh file */
    int N;
    double boxlen;
    double *box1;
    message(rank, "Reading file '%s'\n", pars.InputFilename);
    readFieldFile(&box1, &N, &boxlen, pars.InputFilename);

    /* Open the second mesh file */
    int N2;
    double boxlen2;
    double *box2;
    message(rank, "Reading file '%s'\n", pars.InputFilename2);
    readFieldFile(&box2, &N2, &boxlen2, pars.InputFilename2);

    assert(N == N2);
    assert(boxlen == boxlen2);

    /* Allocate 3D complex arrays */
    fftw_complex *fbox1 = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox2 = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c1 = fftw_plan_dft_r2c_3d(N, N, N, box1, fbox1, FFTW_ESTIMATE);
    fftw_plan r2c2 = fftw_plan_dft_r2c_3d(N, N, N, box2, fbox2, FFTW_ESTIMATE);

    /* Execute and normalize */
    message(rank, "Executing Fourier transforms.\n");
    fft_execute(r2c1);
    fft_execute(r2c2);
    fft_normalize_r2c(fbox1, N, boxlen);
    fft_normalize_r2c(fbox2, N, boxlen);

    /* Destroy the plans */
    fftw_destroy_plan(r2c1);
    fftw_destroy_plan(r2c2);


    if (rank == 0) {
        int bins = pars.PowerSpectrumBins;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));

        message(rank, "Undoing window functions.\n");

        /* Undo the TSC window function */
        undoTSCWindow(fbox1, N, boxlen);
        undoTSCWindow(fbox2, N, boxlen);

        message(rank, "Computing power spectra.\n");

        /* First, the power spectrum of box1 */
        calc_cross_powerspec(N, boxlen, fbox1, fbox1, bins, k_in_bins, power_in_bins, obs_in_bins);

        printf("\n");
        printf("Power spectrum 1:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }

        /* Then, the power spectrum of box2 */
        calc_cross_powerspec(N, boxlen, fbox2, fbox2, bins, k_in_bins, power_in_bins, obs_in_bins);

        printf("\n");
        printf("Power spectrum 2:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }

        /* Then, the cross power spectrum */
        calc_cross_powerspec(N, boxlen, fbox1, fbox2, bins, k_in_bins, power_in_bins, obs_in_bins);

        printf("\n");
        printf("Cross power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }
        
        /* Next, for the total power spectrum, compute the weighted average of the grids */

        /* The indices of the density transfer functions */
        int index_1 = findTitle(ptdat.titles, pars.CrossSpectrumDensity1, ptdat.n_functions);
        int index_2 = findTitle(ptdat.titles, pars.CrossSpectrumDensity2, ptdat.n_functions);

        /* Find the present-day background densities */
        int today_index = ptdat.tau_size - 1; // today corresponds to the last index
        double Omega_1 = ptdat.Omega[ptdat.tau_size * index_1 + today_index];
        double Omega_2 = ptdat.Omega[ptdat.tau_size * index_2 + today_index];

        /* Use the present-day densities as weights */
        double weight_1 = Omega_1 / (Omega_1 + Omega_2);
        double weight_2 = Omega_2 / (Omega_1 + Omega_2);

        message(rank, "Using weights [w_1, w_2] = [%f, %f]\n", weight_1, weight_2);

        /* Compute the weighted average of the two grids, replacing the first */
        /* Also compute the difference fbox2 - fbox1, replacing the second */
        for (int i=0; i<(long long)N*N*(N/2+1); i++) {
            double d1 = fbox1[i];
            double d2 = fbox2[i];
            
            fbox1[i] = weight_1 * d1 + weight_2 * d2;
            fbox2[i] = d2 - d1;
        }
        
        /* Compute the difference power spectrum */
        calc_cross_powerspec(N, boxlen, fbox2, fbox2, bins, k_in_bins, power_in_bins, obs_in_bins);

        printf("\n");
        printf("Difference (delta2 - delta1) power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }

        printf("\n");

        /* Compute the total power spectrum */
        calc_cross_powerspec(N, boxlen, fbox1, fbox1, bins, k_in_bins, power_in_bins, obs_in_bins);

        printf("\n");
        printf("Total power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }

        printf("\n");
    }

    /* Free the grids */
    fftw_free(box1);
    fftw_free(box2);
    fftw_free(fbox1);
    fftw_free(fbox2);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

}
