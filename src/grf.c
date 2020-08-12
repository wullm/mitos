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

    return 0;
}

int enforce_hermiticity(fftw_complex *fbox, int N, int NX, int X0,
                         double boxlen, rng_state *state, MPI_Comm comm) {

    /* The first (k=0) and last (k=N/2+1) planes need hermiticity enforced */

    /* Collect the plane on all nodes */
    fftw_complex *our_slice = fftw_alloc_complex(NX * N);
    fftw_complex *plane = fftw_alloc_complex(N * N);

    /* For both planes */
    for (int z=0; z<=N/2; z+=N/2) { //runs over z=0 and z=N/2

        /* Fill our local slice of the plane */
        for (int x=0; x<NX; x++) {
            for (int y=0; y<N; y++) {
                int id = row_major_half(x,y,z,N);
                our_slice[x*N + y] = fbox[id];
            }
        }

        /* Gather all the slices on all the nodes */
        MPI_Allgather(our_slice, NX * N, MPI_DOUBLE_COMPLEX, plane, NX * N,
                      MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);

        /* Enforce hermiticity: f(k) = f*(-k) */
        for (int x=X0; x<X0 + NX; x++) {
            for (int y=0; y<N; y++) {
                if (x > N/2) continue; //skip the upper half
                if ((x == 0 || x == N/2) && y > N/2) continue; //skip two strips

                int invx = (x > 0) ? N - x : 0;
                int invy = (y > 0) ? N - y : 0;
                int invz = (z > 0) ? N - z : 0; //maps 0->0 and (N/2)->(N/2)

                int id = row_major_half(x-X0,y,z,N);

                /* If the point maps to itself, throw away the imaginary part */
                if (invx == x && invy == y && invz == z) {
                    fbox[id] = creal(fbox[id]);
                } else {
                    /* Otherwise, set it to the conjugate of its mirror point */
                    fbox[id] = conj(plane[invx*N + invy]);
                }
            }
        }

        /* Wait until all the ranks are finished */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Free the memory */
    fftw_free(our_slice);
    fftw_free(plane);

    // fftw_complex *box = fftw_alloc_complex(N * N * (N/2+1));
    //
    // /* Gather all the slices on all the nodes */
    // MPI_Allgather(fbox, NX * N * (N/2+1), MPI_DOUBLE_COMPLEX, box, NX * N * (N/2+1),
    //               MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    //
    // /* Check that it is hermitian */
    // for (int x=0; x<N; x++) {
    //     for (int y=0; y<N; y++) {
    //         for (int z=0; z<=N/2; z+=N/2) { //loops over z=0, N/2 only
    //             int invx = (x > 0) ? N - x : 0;
    //             int invy = (y > 0) ? N - y : 0;
    //             int invz = (z > 0) ? N - z : 0;
    //
    //             int id = row_major_half(x,y,z,N);
    //             int invid = row_major_half(invx,invy,invz,N);
    //
    //             if (box[id] != conj(box[invid]))
    //             printf("Not Hermitian at %d %d %d\n", x, y, z);
    //         }
    //     }
    // }
    //
    // fftw_free(box);

    return 0;
}
