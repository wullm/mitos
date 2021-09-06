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
#include <math.h>
#include <hdf5.h>
#include <fftw3.h>
#include <sys/time.h>

#include "../../include/mitos.h"

#define outname(s,x) sprintf(s, "%s/%s", pars.OutputDirectory, x);
#define printheader(s) printf("\n%s%s%s\n", TXT_BLUE, s, TXT_RESET);

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printheader("Mitos Initial Condition Generator");
    printf("The parameter file is '%s'\n", fname);

    /* Timer */
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    /* Mitos structuress */
    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;
    struct perturb_data ptdat;
    struct perturb_spline spline;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);

    printf("The output directory is '%s'.\n", pars.OutputDirectory);
    printf("Rendering box for '%s'.\n", pars.Name);

    /* Read out particle types from the parameter file */
    readTypes(&pars, &types, fname);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat, pars.PerturbFile);
    
    if (pars.MergeDarkMatterBaryons) {
        printf("Merging cdm & baryon transfer functions, replacing cdm./n");

        /* The indices of the density transfer functions */
        int index_cdm = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
        int index_b = findTitle(ptdat.titles, "d_b", ptdat.n_functions);

        /* Find the present-day background densities */
        int today_index = ptdat.tau_size - 1; // today corresponds to the last index
        double Omega_cdm = ptdat.Omega[ptdat.tau_size * index_cdm + today_index];
        double Omega_b = ptdat.Omega[ptdat.tau_size * index_b + today_index];

        /* Use the present-day densities as weights */
        double weight_cdm = Omega_cdm / (Omega_cdm + Omega_b);
        double weight_b = Omega_b / (Omega_cdm + Omega_b);

        printf("Using weights [w_cdm, w_b] = [%f, %f]\n", weight_cdm, weight_b);

        /* Merge the density & velocity transfer runctions, replacing cdm */
        mergeTransferFunctions(&ptdat, "d_cdm", "d_b", weight_cdm, weight_b);
        mergeTransferFunctions(&ptdat, "t_cdm", "t_b", weight_cdm, weight_b);
        /* Merge the background densities, replacing cdm */
        mergeBackgroundDensities(&ptdat, "d_cdm", "d_b", 1.0, 1.0); //replace with sum
    }

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Determine the starting conformal time */
    cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);

    /* Find the interpolation index along the time dimension */
    double log_tau = cosmo.log_tau_ini; //log of conformal time
    int tau_index; //greatest lower bound bin index
    double u_tau; //spacing between subsequent bins
    perturbSplineFindTau(&spline, log_tau, &tau_index, &u_tau);

    printheader("Settings");
    printf("Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
    printf("\n");

    /* We will read the input into this array */
    double *rho;

    /* Filename of the density grid (should have been stored earlier) */
    printf("Reading input array '%s'.\n", pars.InputFilename);

    /* Read the density field from file */
    int N;
    double boxlen;
    readFieldFile(&rho, &N, &boxlen, pars.InputFilename);

    /* Allocate 3D complex array */
    fftw_complex *fbox = (fftw_complex*) fftw_malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, rho, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, rho, FFTW_ESTIMATE);

    /* Execute and normalize */
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* What transfer function should we apply? */
    int index_src = findTitle(ptdat.titles, "d_ncdm[0]", ptdat.n_functions);

    /* Package the spline parameters */
    struct spline_params sp = {&spline, index_src, tau_index, u_tau};

    /* Apply the transfer function to fbox */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_transfer_function, &sp);

    sp.index_src = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_transfer_function, &sp);

    /* Execute and normalize */
    fft_execute(c2r);
    fft_normalize_c2r(rho, N, boxlen);

    /* Export the real box */
    writeFieldFile(rho, N, boxlen, pars.OutputFilename);
    printf("Resulting field exported to '%s'.\n", pars.OutputFilename);


    /* Free up memory */
    free(rho);
    fftw_destroy_plan(r2c);

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
