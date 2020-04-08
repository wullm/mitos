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
#include <string.h>
#include <math.h>
#include "../include/particle_types.h"

int readTypes(struct params *pars, struct particle_type **tps, const char *fname) {
    /* We need look for no more than this many particle types */
    int max_num = pars->MaxParticleTypes;

    /* Allocate memory for the particle types */
    *tps = malloc(max_num * sizeof(struct particle_type));

    /* Read out the particle types */
    int num = 0;
    for (int i=0; i<max_num; i++) {
        char seek_str[20];
        char identifier[20];
        sprintf(seek_str, "ParticleType_%d", i);
        ini_gets(seek_str, "Identifier", "", identifier, 20, fname);

        /* Have we found a non-empty identifier? */
        if (identifier[0] != '\0') {
            struct particle_type *tp = (*tps) + num;

            tp->Identifier = malloc(strlen(identifier)+1);
            tp->ExportName = malloc(DEFAULT_STRING_LENGTH);
            ini_gets(seek_str, "Identifier", "", tp->Identifier, 20, fname);
            ini_gets(seek_str, "ExportName", "", tp->ExportName, 20, fname);
            tp->Omega = ini_getd(seek_str, "Omega", 1.0, fname);
            tp->Mass = ini_getd(seek_str, "Mass", 1.0, fname);
            tp->TotalNumber = ini_getl(seek_str, "TotalNumber", 0, fname);
            tp->CubeRootNumber = ini_getl(seek_str, "CubeRootNumber", 0, fname);
            tp->Chunks = ini_getl(seek_str, "Chunks", 0, fname);
            tp->ChunkSize = ini_getl(seek_str, "ChunkSize", 0, fname);

            /* Further strings */
            tp->TransferFunctionDensity = malloc(20);
            ini_gets(seek_str, "TransferFunctionDensity", "", tp->TransferFunctionDensity, 20, fname);


            /* Infer total number from cube root number or vice versa */
            if (tp->TotalNumber == 0 && tp->CubeRootNumber > 0) {
                int crn = tp->CubeRootNumber;
                tp->TotalNumber = crn * crn * crn;
            } else if (tp->TotalNumber > 0) {
                tp->CubeRootNumber = ceil(cbrt((double)tp->TotalNumber));
            }

            /* Make sure that Chunks and ChunkSize match */
            if (tp->Chunks == 0 && tp->ChunkSize > 0) {
                tp->Chunks = ceil((double) tp->TotalNumber / tp->ChunkSize);
            } else if (tp->Chunks > 0 && tp->ChunkSize == 0) {
                tp->ChunkSize = ceil((double) tp->TotalNumber / tp->Chunks);
            } else {
                tp->Chunks = 1;
                tp->ChunkSize = tp->TotalNumber;
            }

            /* Infer Omega from Mass or vice versa */
            //To be implemented

            num++;
        }
    }

    pars->NumParticleTypes = num;

    return 0;
}

int cleanTypes(struct params *pars, struct particle_type **tps) {
    for (int i=0; i<pars->NumParticleTypes; i++) {
        struct particle_type *tp = *(tps) + i;
        free(tp->Identifier);
        free(tp->ExportName);
    }
    free(*tps);
    return 0;
}
