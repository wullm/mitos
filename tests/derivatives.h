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

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

/* First order derivatives with five-point stencil */
void compute_derivative_x(double *d_dx, const double *in, int N, double len);
void compute_derivative_y(double *d_dy, const double *in, int N, double len);
void compute_derivative_z(double *d_dz, const double *in, int N, double len);

/* Second order derivatives with five-point stencil */
void compute_derivative_xx(double *d_dx, const double *in, int N, double len);
void compute_derivative_yy(double *d_dy, const double *in, int N, double len);
void compute_derivative_zz(double *d_dz, const double *in, int N, double len);

#endif
