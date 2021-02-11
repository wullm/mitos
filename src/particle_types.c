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
#include <string.h>
#include <math.h>
#include "../include/particle_types.h"
#include "../include/titles.h"
#include "../include/message.h"

int readTypes(struct params *pars, struct particle_type **tps, const char *fname) {
    /* We need look for no more than this many particle types */
    int max_num = pars->MaxParticleTypes;

    /* Allocate memory for the particle types */
    *tps = malloc(max_num * sizeof(struct particle_type));

    /* Grand counter of all particles */
    long long int grand_counter = 0;

    /* Read out the particle types */
    int num = 0;
    for (int i=0; i<max_num; i++) {
        char seek_str[40];
        char identifier[40];
        sprintf(seek_str, "ParticleType_%d", i);
        ini_gets(seek_str, "Identifier", "", identifier, 40, fname);

        /* Have we found a non-empty identifier? */
        if (identifier[0] != '\0') {
            struct particle_type *tp = (*tps) + num;

            tp->Identifier = malloc(strlen(identifier)+1);
            tp->ExportName = malloc(DEFAULT_STRING_LENGTH);
            ini_gets(seek_str, "Identifier", "", tp->Identifier, 20, fname);
            ini_gets(seek_str, "ExportName", "", tp->ExportName, 20, fname);
            // tp->Omega = ini_getd(seek_str, "Omega", 1.0, fname);
            // tp->Mass = ini_getd(seek_str, "Mass", 1.0, fname);
            tp->Multiplicity = ini_getd(seek_str, "Multiplicity", 1.0, fname);
            tp->TotalNumber = ini_getl(seek_str, "TotalNumber", 0, fname);
            tp->CubeRootNumber = ini_getl(seek_str, "CubeRootNumber", 0, fname);
            tp->Chunks = ini_getl(seek_str, "Chunks", 0, fname);
            tp->ChunkSize = ini_getl(seek_str, "ChunkSize", 0, fname);

            tp->CyclesOfMongeAmpere = ini_getl(seek_str, "CyclesOfMongeAmpere", 0, fname);
            tp->CyclesOfSPT = ini_getl(seek_str, "CyclesOfSPT", 0, fname);

            /* Possible input filenames for density and energy flux fields */
            int len = DEFAULT_STRING_LENGTH;
            tp->InputFilenameDensity = malloc(len);
            tp->InputFilenameVelocity = malloc(len);
            ini_gets(seek_str, "InputFilenameDensity", "", tp->InputFilenameDensity, len, fname);
            ini_gets(seek_str, "InputFilenameVelocity", "", tp->InputFilenameVelocity, len, fname);

            /* Further strings */
            tp->TransferFunctionDensity = malloc(20);
            tp->TransferFunctionVelocity = malloc(20);
            tp->ThermalMotionType = malloc(20);
            ini_gets(seek_str, "TransferFunctionDensity", "", tp->TransferFunctionDensity, 20, fname);
            ini_gets(seek_str, "TransferFunctionVelocity", "", tp->TransferFunctionVelocity, 20, fname);
            ini_gets(seek_str, "ThermalMotionType", "", tp->ThermalMotionType, 20, fname);

            /* Firebolt rejection sampler settings */
            tp->FireboltMaxPerturbation = ini_getd(seek_str, "FireboltMaxPerturbation", 0.01, fname);
            tp->UseFirebolt = ini_getbool(seek_str, "UseFirebolt", 0, fname);

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

            /* Use the grand particle counter + 1 to assign the particle ids */
            tp->FirstID = grand_counter + 1;
            grand_counter += tp->TotalNumber;

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
        free(tp->TransferFunctionDensity);
        free(tp->TransferFunctionVelocity);
        free(tp->ThermalMotionType);
        free(tp->InputFilenameDensity);
        free(tp->InputFilenameVelocity);
    }
    free(*tps);
    return 0;
}

int retrieveDensities(struct params *pars, struct cosmology *cosmo,
                      struct particle_type **tps, struct perturb_data *ptdat) {


    /* The number of time steps in the perturbation data */
    int tau_size = ptdat->tau_size;

    /* The index of the present-day, corresponds to the last index in the array */
    int tau_index = tau_size - 1;

    /* The volume of the simulation box */
    double box_len = pars->BoxLen;
    double box_vol = box_len * box_len * box_len;

    /* For each particle type, fetch the user-defined density function title */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = *tps + pti;
        const char *Identifier = ptype->Identifier;

        /* The user-defined title of the density transfer function */
        const char *title = ptype->TransferFunctionDensity;

        /* Skip if not specified */
        if (strcmp("", title) == 0) continue;

        /* Find the title among the transfer functions */
        int index_src = findTitle(ptdat->titles, title, ptdat->n_functions);
        if (index_src < 0) {
            printf("Error: transfer function '%s' not found (%d).\n", title, index_src);
            return 1;
        }

        /* Find the present-day density, as fraction of the critical density */
        double Omega = ptdat->Omega[tau_size * index_src + tau_index];
        double rho = Omega * cosmo->rho_crit * ptype->Multiplicity;
        double Mass = rho * box_vol / ptype->TotalNumber;

        message(pars->rank, "Particle type '%s' has [Omega, Multiplicity, Mass] \t = " \
                "[%f, %.2f, %f U_M]\n", Identifier, Omega, ptype->Multiplicity, Mass);

        /* Store in the particle-type structure */
        ptype->Omega = Omega;
        ptype->Mass = Mass;
    }

    return 0;
}

int retrieveMicroMasses(struct params *pars, struct cosmology *cosmo,
                        struct particle_type **tps, struct perturb_params *ptpars) {


    /* For each particle type, fetch the user-defined density function title */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = *tps + pti;
        const char *Identifier = ptype->Identifier;

        /* The user-defined title of the density transfer function */
        const char *title = ptype->TransferFunctionDensity;

        /* Skip if not specified */
        if (strcmp("", title) == 0) continue;

        /* Check if it matches the format "d_ncdm[%d]" */
        int n_ncdm = -1;

        for (int i=0; i<ptpars->N_ncdm; i++) {
            char check[20];
            sprintf(check, "d_ncdm[%d]", i);
            if (strcmp(check, title) == 0) {
                n_ncdm = i;
            }
        }

        /* Skip if the match is unsuccessful */
        if (n_ncdm < 0) continue;

        /* Retrieve the microscopic mass in electronvolts */
        ptype->MicroscopicMass_eV = ptpars->M_ncdm_eV[n_ncdm];

        /* Also retrieve the temperature */
        ptype->MicroscopyTemperature = ptpars->T_ncdm[n_ncdm] * ptpars->T_CMB;

        message(pars->rank, "Particle type '%s' has microscopic [M, T] = [%f eV, %f U_T].\n",
                Identifier, ptype->MicroscopicMass_eV, ptype->MicroscopyTemperature);
    }

    return 0;
}

/* Multiple particle_types can map into the same export_group. Here we count
 * the number of particles that go into each export group. */
int fillExportGroups(struct params *pars, struct particle_type **tps, struct export_group **grps) {
    /* We need look for no more than this many export groups */
    const int max_num = pars->NumParticleTypes;

    /* We start with no export groups */
    pars->NumExportGroups = 0;

    /* Allocate memory for the export groups */
    *grps = malloc(max_num * sizeof(struct export_group));

    /* For each user-defined particle type */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = *tps + pti;

        /* Its ExportName */
        const char *ExportName = ptype->ExportName;

        /* Check if there is a group by that name */
        char found = 0;
        for (int i = 0; i < pars->NumExportGroups; i++) {
            struct export_group *grp = *grps + i;

            /* We have a match */
            if (strcmp(ExportName, grp->ExportName) == 0) {
                ptype->PositionInExportGroup = grp->TotalNumber;
                grp->TotalNumber += ptype->TotalNumber;
                found = 1;
            }
        }

        if (!found) {
            /* Create a new export_group */
            struct export_group *grp = *grps + pars->NumExportGroups;
            pars->NumExportGroups++;

            /* Copy over the number of particles and the name */
            grp->TotalNumber = ptype->TotalNumber;
            grp->ExportName = malloc(strlen(ptype->ExportName)+1);
            strcpy(grp->ExportName, ptype->ExportName);

            /* For the Particle Type, record that it comes first in this Group */
            ptype->PositionInExportGroup = 0;
        }
    }

    return 0;
}

int cleanExportGroups(struct params *pars, struct export_group **grps) {
    for (int i = 0; i < pars->NumExportGroups; i++) {
        free((*grps + i)->ExportName);
    }
    free(*grps);

    return 0;
}
