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

/* The .ini parser library is minIni */
#include "../parser/minIni.h"

 struct params {
     /* Random parameters */
     long int Seed;

     /* Box parameters */
     int GridSize;
     double BoxLen;

     /* Simulation parameters */
     char *Name;
     int MaxParticleTypes;
     int NumParticleTypes;
     char *TransferFunctionsFile;
     char *TransferFunctionsFormat;

     /* Output parameters */
     char *OutputDirectory;
 };

 struct units {
     double UnitLengthMetres;
     double UnitTimeSeconds;
     double UnitMassKilogram;
 };

int readParams(struct params *parser, const char *fname);
int readUnits(struct units *us, const char *fname);

#endif
