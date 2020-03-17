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

#include "../include/derivatives.h"
#include "../include/dexm.h"

/* Compute a derivative with O(h^5) error using a five-point stencil */
void compute_derivative_x(double *d_dx, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dx[row_major(x,y,z,N)] = 0.d;
                d_dx[row_major(x,y,z,N)] += (1./12.) * in[row_major(x-2,y,z,N)];
                d_dx[row_major(x,y,z,N)] -= (8./12.) * in[row_major(x-1,y,z,N)];
                d_dx[row_major(x,y,z,N)] += (8./12.) * in[row_major(x+1,y,z,N)];
                d_dx[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x+2,y,z,N)];
                d_dx[row_major(x,y,z,N)] /= h;
            }
        }
    }
}

/* Compute a derivative with O(h^5) error using a five-point stencil */
void compute_derivative_y(double *d_dy, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dy[row_major(x,y,z,N)] = 0.d;
                d_dy[row_major(x,y,z,N)] += (1./12.) * in[row_major(x,y-2,z,N)];
                d_dy[row_major(x,y,z,N)] -= (8./12.) * in[row_major(x,y-1,z,N)];
                d_dy[row_major(x,y,z,N)] += (8./12.) * in[row_major(x,y+1,z,N)];
                d_dy[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y+2,z,N)];
                d_dy[row_major(x,y,z,N)] /= h;
            }
        }
    }
}

/* Compute a derivative with O(h^5) error using a five-point stencil */
void compute_derivative_z(double *d_dz, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dz[row_major(x,y,z,N)] = 0.d;
                d_dz[row_major(x,y,z,N)] += (1./12.) * in[row_major(x,y,z-2,N)];
                d_dz[row_major(x,y,z,N)] -= (8./12.) * in[row_major(x,y,z-1,N)];
                d_dz[row_major(x,y,z,N)] += (8./12.) * in[row_major(x,y,z+1,N)];
                d_dz[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y,z+2,N)];
                d_dz[row_major(x,y,z,N)] /= h;
            }
        }
    }
}



/* Compute second derivative with O(h^4) error using a five-point stencil */
void compute_derivative_xx(double *d_dxx, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dxx[row_major(x,y,z,N)] = 0.d;
                d_dxx[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x-2,y,z,N)];
                d_dxx[row_major(x,y,z,N)] += (16./12.) * in[row_major(x-1,y,z,N)];
                d_dxx[row_major(x,y,z,N)] -= (30./12.) * in[row_major(x,y,z,N)];
                d_dxx[row_major(x,y,z,N)] += (16./12.) * in[row_major(x+1,y,z,N)];
                d_dxx[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x+2,y,z,N)];
                d_dxx[row_major(x,y,z,N)] /= h*h;
            }
        }
    }
}


/* Compute second derivative with O(h^4) error using a five-point stencil */
void compute_derivative_yy(double *d_dyy, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dyy[row_major(x,y,z,N)] = 0.d;
                d_dyy[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y-2,z,N)];
                d_dyy[row_major(x,y,z,N)] += (16./12.) * in[row_major(x,y-1,z,N)];
                d_dyy[row_major(x,y,z,N)] -= (30./12.) * in[row_major(x,y,z,N)];
                d_dyy[row_major(x,y,z,N)] += (16./12.) * in[row_major(x,y+1,z,N)];
                d_dyy[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y+2,z,N)];
                d_dyy[row_major(x,y,z,N)] /= h*h;
            }
        }
    }
}


/* Compute second derivative with O(h^4) error using a five-point stencil */
void compute_derivative_zz(double *d_dzz, const double *in, int N, double len) {
    const double h = len/N;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                d_dzz[row_major(x,y,z,N)] = 0.d;
                d_dzz[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y,z-2,N)];
                d_dzz[row_major(x,y,z,N)] += (16./12.) * in[row_major(x,y,z-1,N)];
                d_dzz[row_major(x,y,z,N)] -= (30./12.) * in[row_major(x,y,z,N)];
                d_dzz[row_major(x,y,z,N)] += (16./12.) * in[row_major(x,y,z+1,N)];
                d_dzz[row_major(x,y,z,N)] -= (1./12.) * in[row_major(x,y,z+2,N)];
                d_dzz[row_major(x,y,z,N)] /= h*h;
            }
        }
    }
}
