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

#ifndef ELPT_H
#define ELPT_H

#include <fftw3.h>
#include "input.h"

#define ELPT_BASENAME "elpt"
#define ELPT_RHO "rho"
#define ELPT_RESID "resid"
#define ELPT_DPHI "dphi"
#define ELPT_PHI_RESID "phi_resid"

int elptChunked(double *f, int N, double boxlen, int cycles, char *basename, char *fname);

#endif
