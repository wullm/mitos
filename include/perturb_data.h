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

#ifndef PERTURB_DATA_H
#define PERTURB_DATA_H

#include "input.h"

/* Data structure containing all cosmological perturbation transfer functions
 * T(k, log_tau) as a function of wavenumber and logarithm of conformal time.
 */
struct perturb_data {
    /* Number of wavenumbers */
    int k_size;
    /* Number of time bins */
    int tau_size;
    /* Number of transfer functions */
    int n_functions;
    /* The array of transfer functions (k_size * tau_size * n_functions) */
    double *delta;
    /* Vector of wavenumbers (k_size) */
    double *k;
    /* Vector of logarithmic conformal times (tau_size) */
    double *log_tau;
    /* Vector of corresponding redshifts (tau_size) */
    double *redshift;
    /* Titles of the transfer functions */
    char **titles;
    /* Vector of background densities (tau_size * n_functions) */
    double *Omega;
};

/* Read the perturbation data from file */
int readPerturb(struct params *pars, struct units *us, struct perturb_data *pt);

/* Clean up the memory */
int cleanPerturb(struct perturb_data *pt);

/* Unit conversion factor for transfer functions, depending on the title. */
double unitConversionFactor(const char *title, double unit_length_factor,
                            double unit_time_factor);

#endif
