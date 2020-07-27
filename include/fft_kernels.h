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

#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H

#include "primordial.h"
#include "perturb_spline.h"

typedef void (*kernel_func)(struct kernel *the_kernel);

static inline void kernel_inv_poisson(struct kernel *the_kernel) {
    double k = the_kernel->k;
    the_kernel->kern = (k > 0) ? -1.0/k/k : 1.0;
}

static inline void kernel_dx(struct kernel *the_kernel) {
    double kx = the_kernel->kx;
    the_kernel->kern = I*kx;
}

static inline void kernel_dy(struct kernel *the_kernel) {
    double ky = the_kernel->ky;
    the_kernel->kern = I*ky;
}

static inline void kernel_dz(struct kernel *the_kernel) {
    double kz = the_kernel->kz;
    the_kernel->kern = I*kz;
}

static inline void kernel_power_no_transfer(struct kernel *the_kernel) {
    const struct cosmology *cosmo = (const struct cosmology *) the_kernel->params;
    double k = the_kernel->k;
    double kern = sqrt(primordialPower(k, cosmo));
    the_kernel->kern = kern;
}

struct spline_params {
    const struct perturb_spline *spline;
    int index_src; //index of the source function
    int tau_index; //index along the time direction
    double u_tau; //spacing between nearest indices in the time direction
};

/* Multiply by the transfer function */
static inline void kernel_transfer_function(struct kernel *the_kernel) {
    double k = the_kernel->k;

    if (k == 0) {
        /* Ignore the DC mode */
        the_kernel->kern = 0.f;
    } else {
        /* Unpack the spline parameters */
        struct spline_params *sp = (struct spline_params *) the_kernel->params;
        const struct perturb_spline *spline = sp->spline;
        int index_src = sp->index_src;

        /* Retrieve the tau index and spacing (same for the entire grid) */
        int tau_index = sp->tau_index;
        double u_tau = sp->u_tau;

        /* Find the k index and spacing for this wavevector */
        int k_index;
        double u_k;
        perturbSplineFindK(sp->spline, k, &k_index, &u_k);

        /* Evaluate the transfer function for this tau and k */
        the_kernel->kern = perturbSplineInterp(spline, k_index, tau_index, u_k, u_tau, index_src);
    }
}

/* Divide by the transfer function */
static inline void kernel_inv_transfer_function(struct kernel *the_kernel) {
    double k = the_kernel->k;

    if (k == 0) {
        /* Ignore the DC mode */
        the_kernel->kern = 0.f;
    } else {
        /* Unpack the spline parameters */
        struct spline_params *sp = (struct spline_params *) the_kernel->params;
        const struct perturb_spline *spline = sp->spline;
        int index_src = sp->index_src;

        /* Retrieve the tau index and spacing (same for the entire grid) */
        int tau_index = sp->tau_index;
        double u_tau = sp->u_tau;

        /* Find the k index and spacing for this wavevector */
        int k_index;
        double u_k;
        perturbSplineFindK(sp->spline, k, &k_index, &u_k);

        /* Evaluate the transfer function for this tau and k */
        the_kernel->kern = 1.0 / perturbSplineInterp(spline, k_index, tau_index, u_k, u_tau, index_src);
    }
}



/* Sinc function */
inline double sinc(double x) { return x == 0 ? 1. : sin(x) / x; }

struct Hermite_kern_params {
    int order;
    int N;
    double boxlen;
};


/* Support for undoing the CIC (order = 2), TSC (order = 3), and higher Hermite
 * polynonial window functions. The order is passed as paramater to the kernel. */
static inline void kernel_undo_Hermite_window(struct kernel *the_kernel) {

    if (the_kernel->k == 0) {
        the_kernel->kern = 1.0;
    } else {
        /* Unpack the parameters */
        struct Hermite_kern_params *p = (struct Hermite_kern_params *) the_kernel->params;
        int order = p->order;

        double kx = the_kernel->kx;
        double ky = the_kernel->ky;
        double kz = the_kernel->kz;

        /* The Hermite Window function in Fourier space */
        double W_x = sinc(0.5 * kx * p->boxlen / p->N);
        double W_y = sinc(0.5 * ky * p->boxlen / p->N);
        double W_z = sinc(0.5 * kz * p->boxlen / p->N);
        double W = pow(W_x * W_y * W_z, order);

        the_kernel->kern = 1.0 / W;
    }
}

#endif
