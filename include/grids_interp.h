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

#ifndef GRIDS_INTERP_H
#define GRIDS_INTERP_H

#include "fft.h"
#include "distributed_grid.h"

/* Interpolation methods for contiguous arrays */
double gridNGP(const double *box, int N, double boxlen, double x, double y, double z);
double gridCIC(const double *box, int N, double boxlen, double x, double y, double z);
double gridTSC(const double *box, int N, double boxlen, double x, double y, double z);
double gridPCS(const double *box, int N, double boxlen, double x, double y, double z);

/* Interpolation methods for distributed grids */
double access_grid(struct left_right_slice *lrs, int iX, int iY, int iZ, int N);
double gridNGP_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N);
double gridCIC_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N);
double gridTSC_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N);
double gridPCS_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N);

/* Apply Fourier kernels to undo the window functions */
int undoNGPWindow(fftw_complex *farr, int N, double boxlen);
int undoCICWindow(fftw_complex *farr, int N, double boxlen);
int undoTSCWindow(fftw_complex *farr, int N, double boxlen);
int undoPCSWindow(fftw_complex *farr, int N, double boxlen);

/* Apply Fourier kernels to undo the window functions */
int undoNGPWindowFloat(fftwf_complex *farr, int N, double boxlen);
int undoCICWindowFloat(fftwf_complex *farr, int N, double boxlen);
int undoTSCWindowFloat(fftwf_complex *farr, int N, double boxlen);
int undoPCSWindowFloat(fftwf_complex *farr, int N, double boxlen);

#endif
