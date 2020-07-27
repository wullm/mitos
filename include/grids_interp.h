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

double gridNGP(const double *box, int N, double boxlen, double x, double y, double z);
double gridCIC(const double *box, int N, double boxlen, double x, double y, double z);
double gridTSC(const double *box, int N, double boxlen, double x, double y, double z);
double gridPCS(const double *box, int N, double boxlen, double x, double y, double z);

int undoNGPWindow(fftw_complex *farr, int N, double boxlen);
int undoCICWindow(fftw_complex *farr, int N, double boxlen);
int undoTSCWindow(fftw_complex *farr, int N, double boxlen);
int undoPCSWindow(fftw_complex *farr, int N, double boxlen);

#endif
