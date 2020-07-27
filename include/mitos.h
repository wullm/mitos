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

#ifndef DEXM_H
#define DEXM_H

#include "input.h"
#include "output.h"
#include "header.h"
#include "random.h"
#include "fft.h"
#include "grf.h"
#include "fft_kernels.h"
#include "particle_types.h"
#include "titles.h"
#include "particle.h"
#include "calc_powerspec.h"
#include "primordial.h"
#include "generate_grids.h"
#include "shrink_grids.h"
#include "poisson.h"
#include "elpt.h"
#include "grids_interp.h"
#include "perturb_data.h"
#include "perturb_spline.h"

#define TXT_RED "\033[31;1m"
#define TXT_GREEN "\033[32;1m"
#define TXT_BLUE "\033[34;1m"
#define TXT_RESET "\033[0m"

#define GRID_NAME_GAUSSIAN "gaussian_pure"      // the Gaussian random field
#define GRID_NAME_GAUSSIAN_SMALL "gaussian_small" // a smaller copy of the GRF 
#define GRID_NAME_DENSITY "density"             // energy density
#define GRID_NAME_THETA "theta"                 // energy flux theta
#define GRID_NAME_POTENTIAL "potential"         // Newtonian potential
#define GRID_NAME_DISPLACEMENT "displacement"   // displacement_{x,y,z} fields
#define GRID_NAME_VELOCITY "velocity"           // velocity_{x,y,z} fields
#define GRID_NAME_THETA_POTENTIAL "theta_potential" //-theta/k^2

#endif
