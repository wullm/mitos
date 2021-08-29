/*******************************************************************************
 * This file is part of Mitos.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
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

    /* Read options */
    const char *fname = argv[1];
    printf("The parameter file is %s\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);

    /* The size of the density grid that we will create */
    int N;
    double boxlen;
    double *box;

    /* Read the box */
    printf("Reading field file from '%s'.\n", pars.InputFilename);
    int err = readFieldFile(&box, &N, &boxlen, pars.InputFilename);
    if (err > 0) {
        printf("Error reading file '%s'\n.", pars.InputFilename);
        exit(1);
    }


    printf("BoxSize = %g U_L.\n", boxlen);
    printf("N = %d.\n", N);
    printf("\n");

    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fft_execute(r2c);

    int bins = pars.PowerSpectrumBins;
    double *k_in_bins = malloc(bins * sizeof(double));
    double *power_in_bins = malloc(bins * sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));

    /* Undo the TSC window function */
    undoTSCWindow(fbox, N, boxlen);

    /* Calculate the power spectrum, because it is cheap */
    printf("Computing the power spectrum.\n");
    calc_cross_powerspec(N, boxlen, fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

    /* Check that it is right */
    printf("\n");
    printf("k Pk\n");
    for (int i=0; i<bins; i++) {
        if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

        /* The power we observe */
        double k = k_in_bins[i];
        double Pk = power_in_bins[i];

        printf("%.10g %.10g\n", k, Pk);
    }
    printf("\n");

    /* Free the power spectrum arrays */
    free(k_in_bins);
    free(power_in_bins);
    free(obs_in_bins);

    /* Parse bispectrum parameters */
    int bispectrum_type = pars.BispectrumType;
    double theta = pars.BispectrumAngle; // used for type = 0
    double fixed_k3 = pars.BispectrumMode; // used for type = 1
    int bins3 = pars.BispectrumSecondBins; //number of bins for k3
    if (bins3 == 0) {
        bins3 = bins;
    }

    if (bispectrum_type == 0) {
        printf("Computing the bispectrum in angle mode.\n");
        printf("Running with theta = %g (= %g pi).\n", theta, theta / M_PI);
        printf("Running with %d bins for k3.\n", bins3);
    } else if (bispectrum_type == 1) {
        printf("Computing the bispectrum in fixed mode.\n");
        printf("Running with fixed k3 = %g U_L^-1.\n", fixed_k3);
        printf("Running with %d bins for k3.\n", bins3);
    } else {
        printf("ERROR: Unknown bispectrum type.\n");
        exit(1);
    }
    printf("\n");

    double *k1_in_bins = malloc(bins * sizeof(double));
    double *k2_in_bins = malloc(bins * sizeof(double));
    double *k3_in_bins = malloc(bins * sizeof(double));
    double *bispectrum = malloc(bins * sizeof(double));
    double *bin_volume = malloc(bins * sizeof(double));

    /* Compute the bispectrum */
    calc_bispectrum(N, boxlen, fbox, bins, bins3, k1_in_bins, k2_in_bins, k3_in_bins, bispectrum, 0, bispectrum_type, theta, fixed_k3);

    printf("\n");
    printf("Computing the bin volumes.\n");
    printf("\n");

    /* Compute the normalizing volume */
    calc_bispectrum(N, boxlen, NULL, bins, bins3, k1_in_bins, k2_in_bins, k3_in_bins, bin_volume, 1, bispectrum_type, theta, fixed_k3);

    printf("\n");
    printf("Normalzing the bispectrum.\n");
    printf("\n");

    /* Make the bispectrum dimensionful (units of Vol^2) */
    const double boxvol = boxlen * boxlen * boxlen;
    const double N3 = (double) N * N * N;
    const double norm = boxvol * boxvol / (N3 * N3 * N3);

    for (int i=0; i<bins; i++) {
        bispectrum[i] *= norm;
        bispectrum[i] /= bin_volume[i];
    }

    /* Print the bispectrum */
    printf("\n");
    printf("k1 k2 k3 Bk bin_vol\n");
    for (int i=0; i<bins; i++) {
        if (bin_volume[i] == 0.0) continue; //skip empty bins

        printf("%.10g %.10g %.10g %g %g\n", k1_in_bins[i], k2_in_bins[i], k3_in_bins[i], bispectrum[i], bin_volume[i]);
    }

    printf(" ============ \n");

    /* Print another copy of the bispectrum, skipping negative values */
    printf("k1 k2 k3 Bk bin_vol\n");
    for (int i=0; i<bins; i++) {
        if (bin_volume[i] == 0.0) continue; //skip (virtually) empty bins
        if (bispectrum[i] <= 0.0) continue; //skip negative values

        printf("%.10g %.10g %.10g %g %g\n", k1_in_bins[i], k2_in_bins[i], k3_in_bins[i], bispectrum[i], bin_volume[i]);
    }

    /* Free the grids */
    free(box);
    free(fbox);

    /* Free the bins */
    free(k1_in_bins);
    free(k2_in_bins);
    free(k3_in_bins);
    free(bispectrum);
    free(bin_volume);

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
