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

#ifndef PRIMORDIAL_H
#define PRIMORDIAL_H

#include <math.h>
#include "input.h"
#include "fft.h"
#include "transfer_interp.h"

int initPrimordial(const struct params *pars, const struct cosmology *cosmo);
double primordialPower(double k);

static inline double sigma_func(double k) {
    if (k == 0) return 0;

    return sqrt(primordialPower(k)) * tr_func_at_k(k);
}

static inline void fullPowerSpectrumKernel(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double kern = sigma_func(k);
    the_kernel->kern = kern;
}

#endif
