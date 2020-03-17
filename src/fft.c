/*******************************************************************************
 * This file is part of DEXM.
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

#include <math.h>
#include <stdlib.h>

#include "../include/fft.h"

/* Compute the 3D wavevector (kx,ky,kz) and its length k */
void fft_wavevector(int x, int y, int z, int N, double delta_k, double *kx,
                    double *ky, double *kz, double *k) {
    *kx = (x > N/2) ? (x - N)*delta_k : x*delta_k;
    *ky = (y > N/2) ? (y - N)*delta_k : y*delta_k;
    *kz = (z > N/2) ? (z - N)*delta_k : z*delta_k;
    *k = sqrt((*kx)*(*kx) + (*ky)*(*ky) + (*kz)*(*kz));
}

/* Normalize the complex array after transforming to momentum space */
void fft_normalize_r2c(fftw_complex *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                arr[row_major_half(x, y, z, N)][0] *= boxvol/(N*N*N);
                arr[row_major_half(x, y, z, N)][1] *= boxvol/(N*N*N);
            }
        }
    }
}

/* Normalize the real array after transforming to configuration space */
void fft_normalize_c2r(double *arr, int N, double boxlen) {
    const double boxvol = boxlen*boxlen*boxlen;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                arr[row_major(x, y, z, N)] /= boxvol;
            }
        }
    }
}

/* Execute an FFTW plan */
void fft_execute(fftw_plan plan) {
    fftw_execute(plan);
}

void fft_apply_kernel(fftw_complex *write, const fftw_complex *read, int N,
                      double len, double (*kern)(double,double,double,double)) {
    const double dk = 2 * M_PI / len;

    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                const double kernel = kern(k,kx,ky,kz);
                const int id = row_major_half(x,y,z,N);

                write[id][0] = read[id][0] * kernel;
                write[id][1] = read[id][1] * kernel;
            }
        }
    }
}


/* Quick and dirty write binary boxes */
void write_floats(const char *fname, const float *floats, int n) {
  FILE *f = fopen(fname, "wb");
  fwrite(floats, sizeof(float), n, f);
  fclose(f);
}

/* Quick and dirty write binary boxes */
void write_doubles_as_floats(const char *fname, const double *doubles, int n) {
  /* Convert to floats */
  float *floats = (float *)malloc(sizeof(float) * n);
  for (int i = 0; i < n; i++) {
    floats[i] = (float)doubles[i];
  }

  FILE *f = fopen(fname, "wb");
  fwrite(floats, sizeof(float), n, f);
  fclose(f);
  free(floats);
}
