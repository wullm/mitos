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
#include <assert.h>

#include "../include/spt_convolve.h"
#include "../include/fft.h"


/* The SPT second-order Kernel F2 */
double kernelF2(double k1, double k2, double k1k2) {

    if (k1 == 0 || k2 == 0) return 0.d;

    double r = k1k2 / (k1 * k2);
    double rr = r * r;

    return 5./7. + 0.5 * r * (k1/k2 + k2/k1) + 2./7. * rr;
}

int spt_convolve2(int N, double boxlen, fftw_complex *out_box,
                  const fftw_complex *in_box1, const fftw_complex *in_box2) {

    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;

    /* We need to do a double loop */
    double kx,ky,kz,k;
    double k1x,k1y,k1z,k1;
    double k2x,k2y,k2z,k2; //is constrained to be k2 = k - k1
    double k1k2;

    for (int x=0; x<N; x++) {
        printf("%d/%d\n", x, N);
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the output grid wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* The id of the output cell */
                const long long int id = row_major_half(x, y, z, N);
                out_box[id] = 0.d;

                if (k==0) continue; //skip the DC mode

                for (int x1=0; x1<N; x1++) {
                    for (int y1=0; y1<N; y1++) {
                        for (int z1=0; z1<=N/2; z1++) {
                            /* Calculate the first wavevector */
                            fft_wavevector(x1, y1, z1, N, dk, &k1x, &k1y, &k1z, &k1);

                            /* Calculate the second wavevector */
                            k2x = kx - k1x;
                            k2y = ky - k1y;
                            k2z = kz - k1z;
                            k2 = hypot3(k2x, k2y, k2z);
                            /* Inner product */
                            k1k2 = k1x * k2x + k1y * k2y + k1z * k2z;

                            /* Calculate the grid location of this wavevector */
                            int x2 = (k2x < 0) ? round(k2x / dk + N) : round(k2x / dk);
                            int y2 = (k2y < 0) ? round(k2y / dk + N) : round(k2y / dk);
                            int z2 = (k2z < 0) ? round(k2z / dk + N) : round(k2z / dk);

                            /* Compute the kernel */
                            double K = kernelF2(k1, k2, k1k2);

                            // printf("%f %f %f %f\n", k1, k2, k1k2, K);

                            /* The id of the inout cell */
                            const long long int in_id1 = row_major_half(x1, y1, z1, N);
                            const long long int in_id2 = row_major_half(x2, y2, z2, N);

                            /* Add the result */
                            out_box[id] += K * in_box1[in_id1] * in_box2[in_id2] / boxvol;
                        }
                    }
                }
            }
        }
    }

    return 0;
}
