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
#include <gsl/gsl_spline.h>

#include "../include/transfer_interp.h"

/* GSL interpolation objects */
const gsl_interp_type *tr_interp_type;
gsl_interp_accel *tr_k_acc;
gsl_spline *tr_func_spline;

int tr_interp_init(const struct transfer *tr) {
    /* We will use linear interpolation in k space */
    tr_interp_type = gsl_interp_linear;

    /* Allocate memory for the splines */
    tr_func_spline = gsl_spline_alloc(tr_interp_type, tr->nrow);
    /* Note: this only copies the first function from tr->functions */
    gsl_spline_init(tr_func_spline, tr->k, tr->functions[0], tr->nrow);

    /* Allocate memory for the accelerator objects */
    tr_k_acc = gsl_interp_accel_alloc();

  return 0;
}

int tr_interp_switch_func(const struct transfer *tr, int index_func) {
    /* The array tr->functions contains an array of all transfer functions,
    * each of size tr->nrow doubles */
    int chunk_size = tr->nrow;

    /* Copy the desired background function to the spline */
    double *destination = tr_func_spline->y;
    double *source_address = tr->functions[index_func];
    memcpy(destination, source_address, chunk_size * sizeof(double));

    return 0;
}

int tr_interp_free(const struct transfer *tr) {
    /* Done with the GSL interpolation */
    gsl_spline_free(tr_func_spline);
    gsl_interp_accel_free(tr_k_acc);

    return 0;
}

double tr_func_at_k(double k) {
    return gsl_spline_eval(tr_func_spline, k, tr_k_acc);
}
