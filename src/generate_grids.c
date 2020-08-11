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

/* Generate a perturbation theory grid for each particle type by applying the
 * desired transfer function (given by the titles array) to the random phases.
 * The spline struct is used to interpolate the transfer functions.
 */
int generatePerturbationGrids(const struct params *pars, const struct units *us,
                              const struct cosmology *cosmo,
                              const struct perturb_spline *spline,
                              struct particle_type *types, char **titles,
                              const char *grf_fname, const char *grid_name,
                              int N, int NX, int X0, long int block_size,
                              double boxlen, MPI_Comm comm) {

    /* Find the interpolation index along the time dimension */
    double log_tau = cosmo->log_tau_ini; //log of conformal time
    int tau_index; //greatest lower bound bin index
    double u_tau; //spacing between subsequent bins
    perturbSplineFindTau(spline, log_tau, &tau_index, &u_tau);

    /* For each particle type, create the corresponding density field */
    for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;

        /* The user-defined title of the density transfer function */
        const char *title = titles[pti];

        /* Skip if not specified */
        if (strcmp("", title) == 0) continue;

        /* Find the title among the transfer functions */
        int index_src = findTitle(spline->ptdat->titles, title, spline->ptdat->n_functions);
        if (index_src < 0) {
            printf("Error: transfer function '%s' not found (%d).\n", title, index_src);
            return 1;
        }

        /* Create 3D arrays (block_size nominally is NX*N*(N/2+1), but FFTW
         * may sometimes require more memory for intermediate steps. */
        double *box = fftw_alloc_real(2*block_size);
        fftw_complex *fbox = fftw_alloc_complex(block_size);

        /* Load the Gaussian random field */
        int err = readField_MPI(box, N, NX, X0, comm, grf_fname);
        if (err > 0) return err;

        /* Create FFT plans (destroys input) */
        // fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        /* Create MPI FFTW plans */
        fftw_plan r2c_mpi = fftw_mpi_plan_dft_r2c_3d(N, N, N, box, fbox, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
        fftw_plan c2r_mpi = fftw_mpi_plan_dft_c2r_3d(N, N, N, fbox, box, MPI_COMM_WORLD, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        /* Execute and normalize */
        fft_execute(r2c_mpi);
        fft_normalize_r2c(fbox, N, NX, X0, boxlen);
        fftw_destroy_plan(r2c_mpi);

        /* Package the spline parameters */
        struct spline_params sp = {spline, index_src, tau_index, u_tau};

        /* Apply the transfer function to fbox */
        fft_apply_kernel(fbox, fbox, N, NX, X0, boxlen, kernel_transfer_function, &sp);

        /* Execute and normalize */
        fft_execute(c2r_mpi);
        fft_normalize_c2r(box, N, NX, X0, boxlen);
        fftw_destroy_plan(c2r_mpi);

        /* Export the real box */
        char dbox_fname[DEFAULT_STRING_LENGTH];
        sprintf(dbox_fname, "%s/%s_%s%s", pars->OutputDirectory, grid_name, Identifier, ".hdf5");
        err = writeFieldFile_MPI(box, N, NX, X0, boxlen, MPI_COMM_WORLD, dbox_fname);
        if (err > 0) return err;
        printf("Perturbation field '%s' exported to '%s'.\n", title, dbox_fname);

        // fft_c2r_export_and_free(fbox, N, boxlen, dbox_fname); //frees fbox

        /* Free memory */
        fftw_free(fbox);
        fftw_free(box);
    }

    return 0;
}
