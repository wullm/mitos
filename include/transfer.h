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

#ifndef TRANSFER_H
#define TRANSFER_H

#include "input.h"

struct transfer {
    float *k;
    float **functions;
    char **titles;
    long int nrow;
    int ncol;
};

enum transfer_format {
    Plain,
    CLASS
};

int readTransfers(const struct params *pars, const struct units *us,
                  const struct cosmology *cosmo, struct transfer *trs);
int cleanTransfers(struct transfer *trs);

#endif
