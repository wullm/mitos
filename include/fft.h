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

#ifndef FFT_H
#define FFT_H

#include <fftw3.h>

static inline int row_major(int i, int j, int k, int N) {
    return i*N*N + j*N + k;
}

static inline int row_major_half(int i, int j, int k, int N) {
    return k + (N/2 + 1) * (j + N*i);
}

void fft_wavevector(int x, int y, int z, int N, double delta_k, double *kx,
                    double *ky, double *kz, double *k);

void fft_normalize_r2c(fftw_complex *arr, int N, double boxlen);
void fft_normalize_c2r(double *arr, int N, double boxlen);

void fft_execute(fftw_plan plan);

/* Some useful I/O functions for debugging */
void write_floats(const char *fname, const float *floats, int n);
void write_doubles_as_floats(const char *fname, const double *doubles, int n);


#endif
