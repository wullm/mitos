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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/generate_grids.h"
#include "../include/mitos.h"

int generatePerturbationGrid(const struct cosmology *cosmo,
                             const struct perturb_spline *spline,
                             struct distributed_grid *grf,
                             struct distributed_grid *grid,
                             const char *transfer_func_title,
                             const char *fname, const double rescale_factor) {

    /* Find the interpolation index along the time dimension */
    double log_tau = cosmo->log_tau_source; //log of conformal time
    int tau_index; //greatest lower bound bin index
    double u_tau; //spacing between subsequent bins
    perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);

    /* Find the title among the transfer functions */
    int index_src = findTitle(spline->ptdat->titles, transfer_func_title, spline->ptdat->n_functions);
    if (index_src < 0) {
        printf("Error: transfer function '%s' not found (%d).\n", transfer_func_title, index_src);
        return 1;
    }

    /* Package the perturbation theory interpolation spline parameters */
    struct spline_params sp = {spline, index_src, tau_index, u_tau};

    /* Apply the transfer function */
    fft_apply_kernel_dg(grid, grf, kernel_transfer_function, &sp);

    /* Multiply by the growth factor ratio if needed */
    if (rescale_factor != 1.0) {
        fft_apply_kernel_dg(grid, grid, kernel_constant, &rescale_factor);
    }

    /* Transform back to configuration space */
    fft_c2r_dg(grid);

    /* Export the real box with the density field */
    int err = writeFieldFile_dg(grid, fname);
    if (err > 0) return err;

    return 0;
}
