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

#ifndef GENERATE_GRIDS_H
#define GENERATE_GRIDS_H

#include <fftw3.h>
#include "input.h"
#include "particle_types.h"
#include "perturb_spline.h"
#include "distributed_grid.h"

/* Generate perturbation theory grids by applying transfer functions to
 * the random phases. */
int generatePerturbationGrids(const struct params *pars, const struct units *us,
                              const struct cosmology *cosmo,
                              const struct perturb_spline *spline,
                              struct particle_type *types, char **titles,
                              const char *grf_fname, const char *grid_name,
                              MPI_Comm comm);

int generatePerturbationGrid(const struct cosmology *cosmo,
                             const struct perturb_spline *spline,
                             struct distributed_grid *grf,
                             struct distributed_grid *grid,
                             const char *transfer_func_title,
                             const char *fname);
#endif
