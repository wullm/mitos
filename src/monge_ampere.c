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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_linalg.h>


#include "../include/monge_ampere.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/output.h"
#include "../include/poisson.h"
#include "../include/message.h"

typedef double* dp;

/* Fast 3x3 determinant */
static inline double det3(double *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

/* Solve the Monge-Ampere equation |D.phi| = f using FFT, stopping after a
 * given number of cycles. We require that the density grid has already
 * been Fourier transformed to momentum space. The output will be stored
 * as a complex grid in potential, which should NOT be used in intermediate
 * steps. This allows us to use the same grid for both input and output.
 * The workspace grid should be distinct from the density grid. */
int solveMongeAmpere(struct distributed_grid *potential,
                     struct distributed_grid *density,
                     struct distributed_grid *workspace, int cycles) {

    /* Size of the problem */
    const int N = density->N;
    const int NX = density->NX;
    const long int chunk_size = NX * N * (N + 2); //with padding
    const double boxlen = density->boxlen;
    const MPI_Comm comm = density->comm;

    /* Get the MPI rank */
    int rank;
    MPI_Comm_rank(comm, &rank);

    /* The grids should have the same size and MPI rank distribution */
    assert(potential->NX == density->NX && density->NX == workspace->NX);
    assert(potential->X0 == density->X0 && density->X0 == workspace->X0);
    assert(potential->N == density->N && density->N == workspace->N);
    /* The potential, density, and workspace grids should be distinct memory spaces */
    assert(potential->box != density->box);
    assert(workspace->box != density->box);
    assert(potential->box != workspace->box);

    /* We calculate derivatives using FFT kernels */
    const kernel_func derivatives[] = {kernel_dx, kernel_dy, kernel_dz};
    /* We need xx, xy, xz, yy, yz, zz to compute the Hessian */
    const int index_a[] = {0, 0, 0, 1, 1, 2};
    const int index_b[] = {0, 1, 2, 1, 2, 2};

    /* The density grid should be in momentum space */
    if (density->momentum_space != 1) {
        printf("Error: Density grid is not in momentum space.\n");
        return 1;
    }

    /* Compute initial (Zel'dovich) guess using the inverse Poisson kernel */
    fft_apply_kernel_dg(potential, density, kernel_inv_poisson, NULL);

    /* Transform the density grid back to configuration space */
    fft_r2c_dg(density);

    /* For each M-A cycle */
    for (int ITER = 0; ITER < cycles; ITER++) {

        /* We will need six derivative grids */
        struct distributed_grid *derivative_grids = malloc(6 * sizeof(struct distributed_grid));

        /* Compute the 6 derivative components of the Hessian */
        for (int j=0; j<6; j++) {
            /* Allocate a complex and real grid */
            alloc_local_grid(&derivative_grids[j], N, boxlen, comm);

            /* Compute the derivative d^2 phi / (dx_i dx_j) */
            fft_apply_kernel_dg(&derivative_grids[j], potential, derivatives[index_a[j]], NULL);
            fft_apply_kernel_dg(&derivative_grids[j], &derivative_grids[j], derivatives[index_b[j]], NULL);

            /* Fourier transform to configuration space */
            fft_c2r_dg(&derivative_grids[j]);

            /* Free the complex grid */
            free_local_complex_grid(&derivative_grids[j]);
        }

        /* Accumulate the squared residual error and source */
        double eps = 0.d;
        double norm = 0.d;

        /* At each grid point, compute the determinant and store the residual */
        for (int k=0; k<chunk_size; k++) {
            double d_dxx, d_dyy, d_dzz;
            double d_dxy, d_dxz, d_dyz;

            d_dxx = derivative_grids[0].box[k];
            d_dxy = derivative_grids[1].box[k];
            d_dxz = derivative_grids[2].box[k];
            d_dyy = derivative_grids[3].box[k];
            d_dyz = derivative_grids[4].box[k];
            d_dzz = derivative_grids[5].box[k];

            double M[] = {1+d_dxx, d_dxy, d_dxz,
                          d_dxy, 1+d_dyy, d_dyz,
                          d_dxz, d_dyz, 1+d_dzz};

            double det = det3(M);

            double rho = density->box[k];
            double resid = (1 + rho) - det;

            /* Store the residual */
            workspace->box[k] = resid;

            /* For diagnostics, record the squared residual and source */
            eps += resid * resid;
            norm += rho * rho;
        }

        /* Free the real derivative grids */
        for (int j=0; j<6; j++) {
            free_local_real_grid(&derivative_grids[j]);
        }
        free(derivative_grids);

        /* Solve the Poisson equation, applied just to the residuals */
        solvePoisson_dg(workspace);

        /* Transform the potential grid back to configuration space */
        fft_c2r_dg(potential);

        /* Add the result on top of the potential grid */
        for (int k=0; k<chunk_size; k++) {
            potential->box[k] += workspace->box[k];
        }

        /* Transform the potential grid to momentum space */
        fft_r2c_dg(potential);

        /* Add the squared residuals and densities from all MPI ranks */
        double eps_norm[2] = {eps, norm};
        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, eps_norm, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(eps_norm, eps_norm, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        /* Compute the root mean square residual, normalized by the source grid */
        if (rank == 0) {
            double rms_eps = sqrt((eps_norm[0] / eps_norm[1]) / (N*N*N));
            message(rank, "%03d] Finished MA cycle: eps = %e\n", ITER, rms_eps);
        }
    }

    return 0;
}
