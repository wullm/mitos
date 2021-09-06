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

#ifndef CALC_POWERSPEC_H
#define CALC_POWERSPEC_H

#include <complex.h>
#include <fftw3.h>

void calc_cross_powerspec(int N, double boxlen, const fftw_complex *box1,
                          const fftw_complex *box2, int bins, double *k_in_bins,
                          double *power_in_bins, int *obs_in_bins);
void calc_cross_powerspec_float(int N, double boxlen, const fftwf_complex *box1,
                                const fftwf_complex *box2, int bins, double *k_in_bins,
                                double *power_in_bins, int *obs_in_bins);
void calc_cross_powerspec_2d(int N, double anglesize, const fftw_complex *box1,
                             const fftw_complex *box2, int bins, double *l_in_bins,
                             double *power_in_bins, int *obs_in_bins);
void calc_bispectrum(int N, double boxlen, const fftw_complex *fbox,
                     int bins, int bins3, double *k1_in_bins, double *k2_in_bins,
                     double *k3_in_bins, double *power_in_bins, int unit_mode,
                     int type, double theta, double k3);
#endif
