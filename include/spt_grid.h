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

#ifndef SPT_GRID_H
#define SPT_GRID_H

#include <fftw3.h>

#define SPT_BASENAME "spt"
#define SPT_DENSITY "density"
#define SPT_GRADIENT "gradient"
#define SPT_FLUX "flux"
#define SPT_FLUX_GRADIENT "flux_gradient"
#define SPT_POTENTIAL "potential"
#define SPT_SOURCE "source"
#define SPT_VELOCITY "vel"
#define SPT_TIDAL "tidal"

int sptChunked(double *f, int N, double boxlen, int cycles, char *basename, char *fname);

#endif
