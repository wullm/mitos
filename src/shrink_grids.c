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
int shrinkGrid_dg(struct distributed_grid *out, struct distributed_grid *in) {
    /* Large and small grid sizes */
    int N = in->N; //large
    int M = out->N; //small

    int K = N/M;
    if (K*M != N) {
        printf("Error: M = %d is not a divisor of N = %d.\n", M, N);
        return 1;
    }

    /* Compute small equivalent of large grid MPI slicing */
    out->NX = in->NX / K;
    out->X0 = in->X0 / K;

    if (out->NX * K != in->NX || out->X0 * K != in->X0) {
        printf("Error: small grid does not divide over ranks (%ld, %ld).\n", out->NX, out->X0);
        return 1;
    }

    /* Reset the output array (including padding) */
    memset(out->box, 0, out->NX * M * (M + 2) * sizeof(double));

    /* The zoom factor */
    double factor = 1.0 / (K * K * K);

    /* Create a zoomed out copy of the input array */
    for (int i=in->X0; i<in->X0 + in->NX; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<N; k++) {
                int x = floor(i / K);
                int y = floor(j / K);
                int z = floor(k / K);
                out->box[row_major(x - out->X0, y, z, M)] += factor * in->box[row_major(i - in->X0, j, k, N)];
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

    // for (int x=0; x<M; x++) {
    //     for (int y=0; y<M; y++) {
    //         for (int z=0; z<M; z++) {
    //             double sum = 0;
    //
    //             for (int i=0; i<K; i++) {
    //                 for (int j=0; j<K; j++) {
    //                     for (int k=0; k<K; k++) {
    //                         sum += in[row_major(x*K + i, y*K + j, z*K + k, N)];
    //                     }
    //                 }
    //             }
    //
    //             out[row_major(x, y, z, M)] = sum * factor;
    //         }
    //     }
    // }

    return 0;
}

int shrinkGridExport(int M, char *fname_out, char *fname_in) {
    /* Allocate memory for the output array */
    double *out = malloc(M * M * M * sizeof(double));

    /* Load the input array */
    double *input;
    int N;
    double boxlen;
    readGRF_H5(&input, &N, &boxlen, fname_in);

    /* Shrink the input array */
    int err = shrinkGrid(out, input, M, N);
    if (err > 0) return err;

    /* Write the output array */
    writeGRF_H5(out, M, boxlen, fname_out);

    return 0;
}


// int shrinkGridExport(int M, char *fname_out, char *fname_in);
