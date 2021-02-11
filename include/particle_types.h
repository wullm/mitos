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

#ifndef PARTICLE_TYPES_H
#define PARTICLE_TYPES_H

#include "input.h"
#include "perturb_data.h"

struct particle_type {
    char *Identifier;
    char *ExportName;
    double Omega, Mass;
    double Multiplicity;
    long long int TotalNumber;
    int CubeRootNumber;
    int Chunks;
    int ChunkSize; //except possibly the last chunk
    int CyclesOfMongeAmpere;
    int CyclesOfSPT;

    /* The transfer function titles from CLASS */
    char *TransferFunctionDensity;
    char *TransferFunctionVelocity;

    /* Possible input filenames for density and energy flux fields */
    char *InputFilenameDensity;
    char *InputFilenameVelocity;

    /* Thermal species properties */
    double MicroscopicMass_eV;
    double MicroscopyTemperature;
    char *ThermalMotionType;

    /* Firebolt settings */
    char UseFirebolt;
    double FireboltMaxPerturbation;

    /* Position in the output file and group, assigned automatically */
    long long int FirstID;
    long long int PositionInExportGroup;
};

/* Multiple particle_types can map into the same export_group */
struct export_group {
    char *ExportName;
    long long int TotalNumber;
};

int readTypes(struct params *pars, struct particle_type **tps, const char *fname);
int cleanTypes(struct params *pars, struct particle_type **tps);
int retrieveDensities(struct params *pars, struct cosmology *cosmo,
                      struct particle_type **tps, struct perturb_data *ptdat);
int retrieveMicroMasses(struct params *pars, struct cosmology *cosmo,
                        struct particle_type **tps, struct perturb_params *ptpars);

int fillExportGroups(struct params *pars, struct particle_type **tps, struct export_group **grps);
int cleanExportGroups(struct params *pars, struct export_group **grps);

#endif
