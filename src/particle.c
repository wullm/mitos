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

#include <stdlib.h>
#include <math.h>

#include "../include/particle.h"
#include "../include/dexm.h"

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
                          const struct particle_type *ptype, int chunk) {

    long long int partnum = ptype->TotalNumber;
    long long int chunk_size = ceil((double) partnum / ptype->Chunks);
    int M = ptype->CubeRootNumber;

    /* Throw an error if the particle number is not a cube */
    if (cbrt((double) partnum) != M) {
        printf("ERROR: Number is not a cube; cannot generate particles from grid.\n");
        return 1;
    }

    /* Physical spacing and mass of the particles */
    float len = pars->BoxLen;
    float spacing = len / M;
    float mass = ptype->Mass;

    /* Find where the chunk starts */
    long long start = chunk * chunk_size;
    long long int id = start;
    int x0,y0,z0;
    inverse_row_major(start, &x0, &y0, &z0, M);

    for (int x=x0; x<M; x++) {
        for (int y=y0; y<M; y++) {
            for (int z=z0; z<M; z++) {
                if (id >= start + chunk_size) continue; /* we are done */
                struct particle *part = &(*particles)[id - start];
                part->X = x * spacing;
                part->Y = y * spacing;
                part->Z = z * spacing;
                part->v_X = 0.f;
                part->v_Y = 0.f;
                part->v_Z = 0.f;
                part->mass = mass;
                part->id = id;

                id++;
            }
            z0 = 0; /* rewind back to the start after the first partial row */
        }
        y0 = 0; /* rewind back to the start after the first partial column */
    }

    // for (int i=0; i<chunk_size; i++) {
    //     struct particle *part = &(*particles)[i];
    //     printf("%lld\n", part->id);
    //     part++;
    // }

    return 0;
}
