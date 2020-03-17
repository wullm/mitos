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

#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H


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

#endif
