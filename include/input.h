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

#ifndef INPUT_H
#define INPUT_H

#define DEFAULT_STRING_LENGTH 50

#define MPC_METRES 3.085677581282e22

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

struct params {
    /* Random parameters */
    long int Seed;

    /* Box parameters */
    int GridSize;
    double BoxLen;
    int Splits; //for folding & position dependent power spectra
    int GridX; //for position dependent power spectra
    int GridY; //for position dependent power spectra
    int GridZ; //for position dependent power spectra

    /* Simulation parameters */
    char *Name;
    int MaxParticleTypes;
    int NumParticleTypes;
    char *TransferFunctionsFile;
    char *TransferFunctionsFormat;
    char Homogeneous;

    /* Output parameters */
    char *OutputDirectory;
    char *OutputFilename;

    /* Input parameters */
    char *InputFilename;
    size_t SlabSize;
};

struct units {
    double UnitLengthMetres;
    double UnitTimeSeconds;
    double UnitMassKilogram;

    /* Units for the transfer function input data */
    double TransferUnitLengthMetres;
    int Transfer_hExponent; //1 for h/Mpc; 0 for 1/Mpc
    int Transfer_kExponent; //0 for CLASS; -2 for CAMB/CMBFAST/Eisenstein-Hu
    int Transfer_Sign; //-1 for CLASS; +1 for CAMB/CMBFAST/Eisenstein-Hu
};

struct cosmology {
    double h;
    double n_s;
    double A_s;
    double k_pivot;
};

int readParams(struct params *parser, const char *fname);
int readUnits(struct units *us, const char *fname);
int readCosmology(struct cosmology *cosmo, const char *fname);

int cleanParams(struct params *parser);

int readGRF_H5(double **box, int *N, double *box_len, const char *fname);

#endif
