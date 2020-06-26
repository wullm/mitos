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

#ifndef DENSITY_GRIDS_H
#define DENSITY_GRIDS_H

#include <fftw3.h>
#include "input.h"
#include "particle_types.h"
#include "perturb_spline.h"
#include "transfer.h"

int generateDensityGrids(const struct params *pars, const struct units *us,
                         const struct cosmology *cosmo,
                         const struct transfer *trs,
                         const struct perturb_spline *spline,
                         struct particle_type *types,
                         const fftw_complex *grf);

#endif
