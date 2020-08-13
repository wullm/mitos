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

#include <stdlib.h>
#include <math.h>

#include "../include/particle.h"
#include "../include/mitos.h"

int cleanParticles(struct particle **particles, const struct params *pars,
                   const struct particle_type *ptype) {

    free(*particles);

    return 0;
}

int allocParticles(struct particle **particles, const struct params *pars,
                   const struct particle_type *ptype) {

    long long int partnum = ptype->TotalNumber;
    long long int chunk_size = ceil((double) partnum / ptype->Chunks);
    *particles = malloc(chunk_size * sizeof(struct particle));

    return 0;
}

int genParticles_FromGrid(struct particle **particles, const struct params *pars,
                          const struct units *us, const struct cosmology *cosmo,
                          const struct particle_type *ptype, int chunk,
                          long long int id_first_particle) {

    long long int partnum = ptype->TotalNumber;
    long long int chunk_size = ceil((double) partnum / ptype->Chunks);
    int M = ptype->CubeRootNumber;

    /* Throw an error if the particle number is not a cube */
    if (M*M*M != partnum) {
        printf("ERROR: Number is not a cube; cannot generate particles from grid.\n");
        return 1;
    }

    /* Physical spacing and mass of the particles */
    float len = pars->BoxLen;
    float spacing = len / M;
    float mass = ptype->Mass;

    /* Find where the chunk starts and ends */
    long long start = chunk * chunk_size;
    long long end = (long long) fmin(partnum, start + chunk_size);

    /* Place the particles on a grid */
    for (long long int id = start; id < end; id++) {
        int x,y,z;
        inverse_row_major(id, &x, &y, &z, M);

        struct particle *part = &(*particles)[id - start];
        part->X = x * spacing;
        part->Y = y * spacing;
        part->Z = z * spacing;
        part->v_X = 0.f;
        part->v_Y = 0.f;
        part->v_Z = 0.f;
        part->mass = mass;
        part->id = id + id_first_particle;
    }

    return 0;
}

int genParticlesFromGrid_local(struct particle **particles, const struct params *pars,
                               const struct units *us, const struct cosmology *cosmo,
                               const struct particle_type *ptype, int MX, int X_min,
                               int offset, long long int id_first_particle) {

    long long int partnum = ptype->TotalNumber;
    int M = ptype->CubeRootNumber;

    /* Throw an error if the particle number is not a cube */
    if (M*M*M != partnum) {
        printf("ERROR: Number is not a cube; cannot generate particles from grid.\n");
        return 1;
    }

    /* Physical spacing and mass of the particles */
    float len = pars->BoxLen;
    float spacing = len / M;
    float mass = ptype->Mass;

    int counter = 0;

    for (int x = X_min - offset; x < X_min - offset + MX; x++) {
        for (int y = 0; y < M; y++) {
            for (int z = 0; z < M; z++) {
                long long id = row_major(x, y, z, M) + id_first_particle;

                struct particle *part = &(*particles)[counter];
                part->X = wrap(x, M) * spacing;
                part->Y = wrap(y, M) * spacing;
                part->Z = wrap(z, M) * spacing;
                part->v_X = 0.f;
                part->v_Y = 0.f;
                part->v_Z = 0.f;
                part->mass = mass;
                part->id = id;

                counter++;
            }
        }
    }

    return 0;
}
