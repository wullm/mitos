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

#ifndef DISTRIBUTED_GRID_H
#define DISTRIBUTED_GRID_H

#include <mpi.h>
#include <fftw3-mpi.h>

#define wrap(i,N) ((i)%(N)+(N))%(N)
#define fwrap(x,L) fmod(fmod((x),(L))+(L),(L))

struct distributed_grid {

    /* Global attributes (equal on all MPI ranks) */
    int N;
    double boxlen;
    MPI_Comm comm;

    /* Local attributes */
    long int NX;
    long int X0;
    long int local_size; //number of complex elements = NX * N * (N/2 + 1)

    /* Local portions of the complex and real arrays */
    fftw_complex *fbox;
    double *box;

    /* GLOBAL SIZES:
     * fbox:    N * N * (N/2 + 1)               fftw_complex type
     * box:     N * N * (N + 2)                 double type
     *
     * The real array is padded on the right in the Z-dimension.
     */

    /* LOCAL SIZES:
     * fbox:    NX * N * (N/2 + 1)              fftw_complex type
     * box:     NX * N * (N + 2)                double type
     *
     * The global arrays are sliced along the X-dimension. The local slice
     * corresponds to X0 <= X < X0 + NX.
     */
};

int alloc_local_grid(struct distributed_grid *dg, int N, double boxlen, MPI_Comm comm);
int free_local_grid(struct distributed_grid *dg);

static inline int row_major_dg(int i, int j, int k, struct distributed_grid *dg) {
    /* Wrap global coordinates */
    i = wrap(i,dg->N);
    j = wrap(j,dg->N);
    k = wrap(k,dg->N + 2); //padding

    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0;
    return i*dg->N*(dg->N+2) + j*(dg->N+2) + k;
}

static inline int row_major_half_dg(int i, int j, int k, struct distributed_grid *dg) {
    /* Wrap global coordinates */
    i = wrap(i,dg->N);
    j = wrap(j,dg->N);
    k = wrap(k,dg->N/2 + 1);

    /* Map to local slice (no out of bounds handling) */
    i = i - dg->X0;
    return i*(dg->N/2+1)*dg->N + j*(dg->N/2+1) + k;
}

#endif
