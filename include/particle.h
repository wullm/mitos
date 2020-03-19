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

#ifndef PARTICLE_H
#define PARTICLE_H

#include "dexm.h"

struct particle {
    float X,Y,Z;
    float v_X, v_Y, v_Z;
    float mass;
    long long int id;
};

int allocParticles(struct particle **particles, const struct params *pars,
                   const struct particle_type *ptype);

int cleanParticles(struct particle **particles, const struct params *pars,
                   const struct particle_type *ptype);

int genParticles_FromGrid(struct particle **particles, const struct params *pars,
                          const struct units *us, const struct cosmology *cosmo,
                          const struct particle_type *ptype, int chunk);

#endif
