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

#ifndef ZELDOVICH_H
#define ZELDOVICH_H

#include <fftw3.h>
#include "input.h"
#include "particle_types.h"

/* Solve the Poisson equation D.phi = f using FFT */
int solvePoisson(double *phi, double *f, int N, double boxlen);

/* Solve the Poisson equation for each density grid */
int computePotentialGrids(const struct params *pars, const struct units *us,
                          const struct cosmology *cosmo,
                          struct particle_type *types);
/* Compute derivatives of the potential grids */
int computePotentialDerivatives(const struct params *pars, const struct units *us,
                                const struct cosmology *cosmo,
                                struct particle_type *types);

#endif
