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
    readPerturb(&pars, &us, &ptdat);

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* We will read the input into this array */
    double *rho;

    /* Filename of the density grid (should have been stored earlier) */
    printf("Reading input array '%s'.\n", pars.InputFilename);

    /* Read the density field from file */
    int N;
    double boxlen;
    readFieldFile(&rho, &N, &boxlen, pars.InputFilename);

    /* Allocate 3D complex array */
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, rho, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, rho, FFTW_ESTIMATE);

    /* Execute and normalize */
    fft_execute(r2c);
    fft_normalize_r2c(fbox, N, boxlen);

    /* Normalization of the GRF */
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);

    /* Discard amplitudes, keep the phases, and apply a global normalization */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                double a = fbox[row_major_half(x,y,z,N)][0];
                double b = fbox[row_major_half(x,y,z,N)][1];
                double norm = hypot(a,b);
                if (norm > 0.) {
                    fbox[row_major_half(x,y,z,N)][0] *= sqrt(2.0) * factor / norm;
                    fbox[row_major_half(x,y,z,N)][1] *= sqrt(2.0) * factor / norm;
                }
            }
        }
    }

    /* Apply the bare power spectrum to fbox */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_power_no_transfer, &cosmo);

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
