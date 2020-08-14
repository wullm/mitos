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

// #include <hdf5.h>
#include <stdlib.h>
#include <string.h>

#include "../include/shrink_grids.h"
#include "../include/output.h"
#include "../include/mitos.h"

/* Shrink an N*N*N grid into an M*M*M grid, where M divides N */
int shrinkGrid_dg(double *out, struct distributed_grid *in, int M, int N) {
    int K = N/M;
    if (K*M != N) {
        printf("Error: M = %d is not a divisor of N = %d.\n", M, N);
        return 1;
    }

    /* Reset the output array */
    memset(out, 0, M * M * M * sizeof(double));

    /* The zoom factor */
    double factor = 1.0 / (K * K * K);

    /* Create a zoomed out copy of the input array */
    for (int i=in->X0; i<in->X0 + in->NX; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<N; k++) {
                int x = floor(i / K);
                int y = floor(j / K);
                int z = floor(k / K);
                out[row_major(x, y, z, M)] += factor * in->box[row_major_dg(i, j, k, in)];
            }
        }
    }

    return 0;
}

/* Shrink an N*N*N grid into an M*M*M grid, where M divides N */
int shrinkGrid(double *out, const double *in, int M, int N) {
    int K = N/M;
    if (K*M != N) {
        printf("Error: M = %d is not a divisor of N = %d.\n", M, N);
        return 1;
    }

    /* Reset the output array */
    memset(out, 0, M * M * M * sizeof(double));

    /* The zoom factor */
    double factor = 1.0 / (K * K * K);

    /* Create a zoomed out copy of the input array */
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<N; k++) {
                int x = floor(i / K);
                int y = floor(j / K);
                int z = floor(k / K);
                out[row_major(x, y, z, M)] += factor * in[row_major(i, j, k, N)];
            }
        }
    }

    return 0;
}
