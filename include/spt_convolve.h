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

#ifndef SPT_CONVOLVE_H
#define SPT_CONVOLVE_H

#include <complex.h>
#include <fftw3.h>

int spt_convolve2(int N, double boxlen, fftw_complex *out_box,
                  const fftw_complex *in_box1, const fftw_complex *in_box2);

#endif
