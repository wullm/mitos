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

#include <math.h>
#include "../include/grids_interp.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"

double gridNGP(const double *box, int N, double boxlen, double x, double y, double z) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    return box[row_major(iX, iY, iZ, N)];
}

double gridCIC(const double *box, int N, double boxlen, double x, double y, double z) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.0;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx <= 1 ? 1-xx : 0;
                double part_y = yy <= 1 ? 1-yy : 0;
                double part_z = zz <= 1 ? 1-zz : 0;

                sum += box[row_major(iX+i, iY+j, iZ+k, N)] * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}

double gridTSC(const double *box, int N, double boxlen, double x, double y, double z) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.5;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx < 0.5 ? (0.75-xx*xx)
                                        : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                        : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                        : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);

                sum += box[row_major(iX+i, iY+j, iZ+k, N)] * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}

double gridPCS(const double *box, int N, double boxlen, double x, double y, double z) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with QIP */
    double lookLength = 2;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx < 1.0 ? (4. - 6.*xx*xx + 3.*xx*xx*xx)
                                        : (xx < 2.0 ? (2.-xx)*(2.-xx)*(2.-xx) : 0);
				double part_y = yy < 1.0 ? (4. - 6.*yy*yy + 3.*yy*yy*yy)
                                        : (yy < 2.0 ? (2.-yy)*(2.-yy)*(2.-yy) : 0);
				double part_z = zz < 1.0 ? (4. - 6.*zz*zz + 3.*zz*zz*zz)
                                        : (zz < 2.0 ? (2.-zz)*(2.-zz)*(2.-zz) : 0);

                sum += box[row_major(iX+i, iY+j, iZ+k, N)] * (part_x*part_y*part_z);
            }
        }
    }

    /* Finally, apply the 1/6^3 factor of the Piecewise Cubic Spline */
    sum /= 6 * 6 * 6;

    return sum;
}


/* When the grid is distributed over several MPI nodes, we may need to
 * access cells that lie beyond the range of the current grid. We therefore
 * load slivers on the left and right of the local slice on each node.
 * This methods accesses the cell in the right sliver or slice. */
double access_grid(struct left_right_slice *lrs, int iX, int iY, int iZ, int N) {

    /* Make sure that the coordinates wrap around the box */
    iX = wrap(iX, N);
    iY = wrap(iY, N);
    iZ = wrap(iZ, N); //we only want real data, so skip over the padded rows

    /* The edges to be checked */
    const int lX0 = lrs->left_X0;
    const int cX0 = lrs->local_X0;
    const int rX0 = lrs->right_X0;
    const int lNX = lrs->left_NX;
    const int cNX = lrs->local_NX;
    const int rNX = lrs->right_NX;

    /* Are we in the local slice or should we use the left/right slivers? */
    if (iX >= cX0 && iX < cX0 + cNX) {
        return lrs->local_slice[row_major_padded(iX - cX0, iY, iZ, N)];
    } else if (iX >= lX0 && iX < lX0 + lNX) {
        return lrs->left_slice[row_major_padded(iX - lX0, iY, iZ, N)];
    } else if (iX >= rX0 && iX < rX0 + rNX) {
        return lrs->right_slice[row_major_padded(iX - rX0, iY, iZ, N)];
    } else {
        printf("ERROR: outside of bounds %d %d %d.\n", iX, lX0, rX0 + rNX);
    }

    return 0;
}

/* (Distributed grid version) */
double gridNGP_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    return access_grid(lrs, iX, iY, iZ, N);
}

/* (Distributed grid version) */
double gridCIC_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.0;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx <= 1 ? 1-xx : 0;
                double part_y = yy <= 1 ? 1-yy : 0;
                double part_z = zz <= 1 ? 1-zz : 0;

                sum += access_grid(lrs, iX+i, iY+j, iZ+k, N) * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}

/* (Distributed grid version) */
double gridTSC_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with CIC or TSC */
    double lookLength = 1.5;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx < 0.5 ? (0.75-xx*xx)
                                        : (xx < 1.5 ? 0.5*(1.5-xx)*(1.5-xx) : 0);
				double part_y = yy < 0.5 ? (0.75-yy*yy)
                                        : (yy < 1.5 ? 0.5*(1.5-yy)*(1.5-yy) : 0);
				double part_z = zz < 0.5 ? (0.75-zz*zz)
                                        : (zz < 1.5 ? 0.5*(1.5-zz)*(1.5-zz) : 0);

                sum += access_grid(lrs, iX+i, iY+j, iZ+k, N) * (part_x*part_y*part_z);
            }
        }
    }

    return sum;
}

/* (Distributed grid version) */
double gridPCS_dg(struct left_right_slice *lrs, double x, double y, double z, double boxlen, int N) {
    /* Convert to float grid dimensions */
    double X = x*N/boxlen;
    double Y = y*N/boxlen;
    double Z = z*N/boxlen;

    /* Integer grid position */
    int iX = (int) floor(X);
    int iY = (int) floor(Y);
    int iZ = (int) floor(Z);

    /* Intepolate the necessary fields with QIP */
    double lookLength = 2;
    int lookLftX = (int) floor((X-iX) - lookLength);
    int lookRgtX = (int) floor((X-iX) + lookLength);
    int lookLftY = (int) floor((Y-iY) - lookLength);
    int lookRgtY = (int) floor((Y-iY) + lookLength);
    int lookLftZ = (int) floor((Z-iZ) - lookLength);
    int lookRgtZ = (int) floor((Z-iZ) + lookLength);

    /* Accumulate */
    double sum = 0;
    for (int i=lookLftX; i<=lookRgtX; i++) {
        for (int j=lookLftY; j<=lookRgtY; j++) {
            for (int k=lookLftZ; k<=lookRgtZ; k++) {
                double xx = fabs(X - (iX+i));
                double yy = fabs(Y - (iY+j));
                double zz = fabs(Z - (iZ+k));

                double part_x = xx < 1.0 ? (4. - 6.*xx*xx + 3.*xx*xx*xx)
                                        : (xx < 2.0 ? (2.-xx)*(2.-xx)*(2.-xx) : 0);
				double part_y = yy < 1.0 ? (4. - 6.*yy*yy + 3.*yy*yy*yy)
                                        : (yy < 2.0 ? (2.-yy)*(2.-yy)*(2.-yy) : 0);
				double part_z = zz < 1.0 ? (4. - 6.*zz*zz + 3.*zz*zz*zz)
                                        : (zz < 2.0 ? (2.-zz)*(2.-zz)*(2.-zz) : 0);

                sum += access_grid(lrs, iX+i, iY+j, iZ+k, N) * (part_x*part_y*part_z);
            }
        }
    }

    /* Finally, apply the 1/6^3 factor of the Piecewise Cubic Spline */
    sum /= 6 * 6 * 6;

    return sum;
}

int undoNGPWindow(fftw_complex *farr, int N, double boxlen) {
    /* Package the kernel parameter */
    struct Hermite_kern_params Hkp;
    Hkp.order = 1; //NGP
    Hkp.N = N;
    Hkp.boxlen = boxlen;

    /* Apply the kernel */
    fft_apply_kernel(farr, farr, N, N, 0, boxlen, kernel_undo_Hermite_window, &Hkp);

    return 0;
}


int undoCICWindow(fftw_complex *farr, int N, double boxlen) {
    /* Package the kernel parameter */
    struct Hermite_kern_params Hkp;
    Hkp.order = 2; //CIC
    Hkp.N = N;
    Hkp.boxlen = boxlen;

    /* Apply the kernel */
    fft_apply_kernel(farr, farr, N, N, 0, boxlen, kernel_undo_Hermite_window, &Hkp);

    return 0;
}

int undoTSCWindow(fftw_complex *farr, int N, double boxlen) {
    /* Package the kernel parameter */
    struct Hermite_kern_params Hkp;
    Hkp.order = 3; //TSC
    Hkp.N = N;
    Hkp.boxlen = boxlen;

    /* Apply the kernel */
    fft_apply_kernel(farr, farr, N, N, 0, boxlen, kernel_undo_Hermite_window, &Hkp);

    return 0;
}

int undoPCSWindow(fftw_complex *farr, int N, double boxlen) {
    /* Package the kernel parameter */
    struct Hermite_kern_params Hkp;
    Hkp.order = 4; //PCS
    Hkp.N = N;
    Hkp.boxlen = boxlen;

    /* Apply the kernel */
    fft_apply_kernel(farr, farr, N, N, 0, boxlen, kernel_undo_Hermite_window, &Hkp);

    return 0;
}
