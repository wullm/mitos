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

#include "../include/grf.h"
#include "../include/mitos.h"

#include <math.h>


int generate_complex_grf(fftw_complex *fbox, int N, int NX, int X0,
                         double boxlen, rng_state *state) {
    const double dk = 2 * M_PI / boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);

    /* Refer to fourier.pdf for details. */

    /* Because the Gaussian field is real, the Fourier transform fbox
     * is Hermitian. This can be stored with just N*N*(N/2+1) complex
     * numbers. We loop over x,y in {0, ..., N-1} and z in {0, ..., N/2}.
     */

    double kx,ky,kz,k;
    for (int x=0; x<NX; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x+X0, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Ignore the constant DC mode */
                if (k > 0) {
                    double a = sampleNorm(state) * factor;
                    double b = sampleNorm(state) * factor;
                    fbox[row_major_half(x,y,z,N)] = a + b * I;
                } else {
                    fbox[row_major_half(x,y,z,N)] = 0;
                }
            }
        }
    }

    // /* Enforce hermiticity: f(k) = f*(-k) */
    // for (int x=0; x<N; x++) {
    //     for (int y=0; y<N; y++) {
    //         for (int z=0; z<=N/2; z+=N/2) { //loops over z=0, N/2 only
    //             int invx = (x > 0) ? N - x : 0;
    //             int invy = (y > 0) ? N - y : 0;
    //             int invz = (z > 0) ? N - z : 0;
    //
    //             int id = row_major_half(x,y,z,N);
    //             int invid = row_major_half(invx,invy,invz,N);
    //             fbox[id] =  conj(fbox[invid]);
    //         }
    //     }
    // }

    return 0;
}

int enforce_hermiticity(fftw_complex *fbox, int N, int NX, int X0,
                         double boxlen, rng_state *state, MPI_Comm comm) {
    const double dk = 2 * M_PI / boxlen;
    const double boxvol = boxlen*boxlen*boxlen;
    const double factor = sqrt(boxvol/2);

    /* Fetch the mirror slice to enforce hermiticity: f(k) = f*(-k) */
    fftw_complex *mirror = fftw_alloc_complex(NX * (N/2+1) * 1);


    if (X0 >= N/2) {
        printf("X0=%d\n", X0);
        /* Send */
        for (int x=0; x<NX; x++) {
            for (int y=N/2; y<N; y++) {
                int z = 0;
                int id = row_major_half(x,y,z,N);
                mirror[x*(N/2+1) + y];
            }
        }

        int ranks = N/NX;
        int rank = X0/NX;
        int dest = ranks - rank - 1;
        printf("rank=%d, sending to %d (%d)\n", rank, dest, ranks);
        MPI_Send(mirror, NX*(N/2+1), MPI_DOUBLE, dest, 0, comm);
    } else {
        /* Receive */
        int ranks = N/NX;
        int rank = X0/NX;
        int src = ranks - rank - 1;
        printf("rank=%d, getting from %d (%d)\n", rank, src, ranks);
        MPI_Recv(mirror, NX*(N/2+1), MPI_DOUBLE, src, 0, comm, NULL);

        /* Enforce hermiticity: f(k) = f*(-k) */
        for (int x=0; x<NX; x++) {
            for (int y=0; y<N; y++) {
                int z = 0;

                int invx = (x > 0) ? NX - x : 0;
                int invy = (y > 0) ? N - y : 0;
                int invz = (z > 0) ? N - z : 0;

                int id = row_major_half(x,y,z,N);
                int invid = row_major_half(invx,invy,invz,N);
                fbox[id] =  conj(mirror[invx*(N/2+1) + invy]);
            }
        }
    }

    return 0;
}
