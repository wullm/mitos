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

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);

    /* The size of the density grid that we will create */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Allocate distributed memory arrays (one complex & one real) */
    struct distributed_grid box;
    alloc_local_grid(&box, N, boxlen, MPI_COMM_WORLD);

    /* Allocate other arrays for the operations */
    struct distributed_grid box2, box3, box4;
    alloc_local_grid(&box2, N, boxlen, MPI_COMM_WORLD);
    alloc_local_grid(&box3, N, boxlen, MPI_COMM_WORLD);
    alloc_local_grid(&box4, N, boxlen, MPI_COMM_WORLD);

    /* Read the box */
    int err = readFieldFile_dg(&box, pars.InputFilename);
    if (err > 0) {
        printf("Error reading file '%s'\n.", pars.InputFilename);
        exit(1);
    }

    /* Fourier transform it */
    fft_r2c_dg(&box);

    /* Whiten the field */
    for (int i=0; i<N*N*(N/2+1); i++) {
        box.fbox[i] /= cabs(box.fbox[i]);
    }

    /* Draw random directions on the sphere */
    int points = 250;
    double r = 0;
    double eta2 = 16;

    for (int u=0; u<8; u++) {
        r = (u+1)*5;

        double l_sum = 0.0;
        for (int i=0; i<points; i++) {
            /* Generate a random point on the unit sphere using Gaussians */
            double nx = sampleNorm(&seed);
            double ny = sampleNorm(&seed);
            double nz = sampleNorm(&seed);

            /* And normalize */
            double length = hypot(nx, hypot(ny, nz));
            if (length > 0) {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            /* Apply the eliptic top-hat filter and translation*/
            double elr1[4] = {r*nx, r*ny, r*nz, eta2};
            fft_apply_kernel_dg(&box2, &box, kernel_elliptic_tophat, elr1);
            fft_apply_kernel_dg(&box2, &box2, kernel_translate, elr1);
            double elr2[4] = {-r*nx, -r*ny, -r*nz, eta2};
            fft_apply_kernel_dg(&box3, &box, kernel_elliptic_tophat, elr2);
            fft_apply_kernel_dg(&box3, &box3, kernel_translate, elr2);

            /* Fourier transform them all back */
            fft_c2r_dg(&box2);
            fft_c2r_dg(&box3);

            /* Compute the bispectrum */
            for (int j=0; j<N*N*N; j++) {
                box2.box[j] = box2.box[j] * box3.box[j] * box.box[j];
            }

            // char box_fname[40];
            // sprintf(box_fname, "density_whitened_%d.hdf5", i);
            // writeFieldFile(box2.box, N, boxlen, box_fname);

            /* Fourier transform it again */
            fft_r2c_dg(&box2);

            /* Extract the k=0 mode */
            double a = creal(box2.fbox[0]);
            double b = cimag(box2.fbox[0]);

            l_sum += a;

            // printf("%f %f %f (%e, %e)\n", nx, ny, nz, a, b);
        }

        double l_avg = l_sum / points;

        printf("%e %e\n", r, l_avg);
    }

    /* Fourier transform it back */
    fft_c2r_dg(&box2);

    char box_fname[40];
    sprintf(box_fname, "density_whitened.hdf5");
    writeFieldFile(box2.box, N, boxlen, box_fname);

    /* Free the grids */
    free_local_grid(&box);
    free_local_grid(&box2);
    free_local_grid(&box3);
    free_local_grid(&box4);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
