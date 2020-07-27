#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../include/mitos.h"
#include "derivatives.h"

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
}

static inline void sigma_func(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double kern = exp(-k*k/0.02);
    the_kernel->kern = kern;
}

int main() {
    /* Read parameters */
    const char fname[] = "test_cosmology.ini";
    struct params pars;
    struct units us;

    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Seed the random number generator */
    srand(pars.Seed);

    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Create 3D arrays */
    double *box = (double*) fftw_malloc(N*N*N*sizeof(double));
    double *compare = (double*) fftw_malloc(N*N*N*sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate a Gaussian random field */
    generate_complex_grf(fbox, N, boxlen);
    fft_apply_kernel(fbox, fbox, N, boxlen, sigma_func);

    /* Transform to real configuraiton space */
    fft_execute(c2r);
	fft_normalize_c2r(box,N,boxlen);

    /* Store a deep copy of the GRF in order to compare later */
    memcpy(compare, box, N*N*N*sizeof(double));

    /* Transform back to momentum space */
    fft_execute(r2c);
	fft_normalize_r2c(fbox,N,boxlen);

    /* Solve Poisson's equation (D^2.phi = rho) to find the potential */
    fft_apply_kernel(fbox, fbox, N, boxlen, kernel_inv_poisson);

    /* Next, we will perform some tests by differentiating the potential */

    /* Allocate more memory */
    fftw_complex *fpsi_x = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fpsi_y = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fpsi_z = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Apply derivative kernels */
    fft_apply_kernel(fpsi_x, fbox, N, boxlen, kernel_dx);
    fft_apply_kernel(fpsi_y, fbox, N, boxlen, kernel_dy);
    fft_apply_kernel(fpsi_z, fbox, N, boxlen, kernel_dz);

    /* Transform the primary array to real configuration space */
    fft_execute(c2r);
    fft_normalize_c2r(box,N,boxlen);

    /* Free memory that's no longer needed */
    fftw_free(fbox);
    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    /* Allocate more memory */
    double *psi_x = (double*) fftw_malloc(N*N*N*sizeof(double));
    double *psi_y = (double*) fftw_malloc(N*N*N*sizeof(double));
    double *psi_z = (double*) fftw_malloc(N*N*N*sizeof(double));

    /* Create FFT plans */
    fftw_plan r2c_x = fftw_plan_dft_r2c_3d(N, N, N, psi_x, fpsi_x, FFTW_ESTIMATE);
    fftw_plan r2c_y = fftw_plan_dft_r2c_3d(N, N, N, psi_y, fpsi_y, FFTW_ESTIMATE);
    fftw_plan r2c_z = fftw_plan_dft_r2c_3d(N, N, N, psi_z, fpsi_z, FFTW_ESTIMATE);
    fftw_plan c2r_x = fftw_plan_dft_c2r_3d(N, N, N, fpsi_x, psi_x, FFTW_ESTIMATE);
    fftw_plan c2r_y = fftw_plan_dft_c2r_3d(N, N, N, fpsi_y, psi_y, FFTW_ESTIMATE);
    fftw_plan c2r_z = fftw_plan_dft_c2r_3d(N, N, N, fpsi_z, psi_z, FFTW_ESTIMATE);

    /* Transform to real configuration space */
    fft_execute(c2r_x);
    fft_execute(c2r_y);
    fft_execute(c2r_z);
    fft_normalize_c2r(psi_x,N,boxlen);
    fft_normalize_c2r(psi_y,N,boxlen);
    fft_normalize_c2r(psi_z,N,boxlen);

    /* Allocate memory for derivatives, calculated with a different method */
    double *d_dx = (double*) fftw_malloc(N*N*N*sizeof(double));
    double *d_dy = (double*) fftw_malloc(N*N*N*sizeof(double));
    double *d_dz = (double*) fftw_malloc(N*N*N*sizeof(double));

    /* Calculate derivatives using finite differences */
    compute_derivative_x(d_dx, box, N, boxlen);
    compute_derivative_y(d_dy, box, N, boxlen);
    compute_derivative_z(d_dz, box, N, boxlen);

    /* We should get approximately the same answer, so let's verify. */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                int id = row_major_half(x, y, z, N);
                assert(fabs(d_dx[id] - psi_x[id]) < 1e-3);
                assert(fabs(d_dy[id] - psi_y[id]) < 1e-3);
                assert(fabs(d_dz[id] - psi_z[id]) < 1e-3);
            }
        }
    }

    /* Transform the derivative vector psi back to momentum space */
    fft_execute(r2c_x);
    fft_execute(r2c_y);
    fft_execute(r2c_z);
    fft_normalize_r2c(fpsi_x,N,boxlen);
    fft_normalize_r2c(fpsi_y,N,boxlen);
    fft_normalize_r2c(fpsi_z,N,boxlen);

    /* Apply kernels to differentiate again */
    fft_apply_kernel(fpsi_x, fpsi_x, N, boxlen, kernel_dx);
    fft_apply_kernel(fpsi_y, fpsi_y, N, boxlen, kernel_dy);
    fft_apply_kernel(fpsi_z, fpsi_z, N, boxlen, kernel_dz);

    /* Transform to real configuration space */
    fft_execute(c2r_x);
    fft_execute(c2r_y);
    fft_execute(c2r_z);
    fft_normalize_c2r(psi_x,N,boxlen);
    fft_normalize_c2r(psi_y,N,boxlen);
    fft_normalize_c2r(psi_z,N,boxlen);

    /* Compute second derivatives using finite differences */
    compute_derivative_xx(d_dx, box, N, boxlen);
    compute_derivative_yy(d_dy, box, N, boxlen);
    compute_derivative_zz(d_dz, box, N, boxlen);

    /* We should get approximately the same result, so let's verify. */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                int id = row_major_half(x, y, z, N);
                assert(fabs(d_dx[id] - psi_x[id]) < 1e-4);
                assert(fabs(d_dy[id] - psi_y[id]) < 1e-4);
                assert(fabs(d_dz[id] - psi_z[id]) < 1e-4);
            }
        }
    }

    /* Free memory that's no longer needed */
    fftw_free(d_dx);
    fftw_free(d_dy);
    fftw_free(d_dz);
    fftw_free(fpsi_x);
    fftw_free(fpsi_y);
    fftw_free(fpsi_z);

    /* Add it all back together */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                int id = row_major(x,y,z,N);
                box[id] = psi_x[id] + psi_y[id] + psi_z[id];
            }
        }
    }

    /* We should get back the original GRF, so let's verify. */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<N; z++) {
                int id = row_major_half(x, y, z, N);
                assert(fabs(compare[id] - box[id]) < 1e-10);
            }
        }
    }

    /* Free the remaining memory */
    fftw_free(box);
    fftw_free(compare);
    fftw_free(psi_x);
    fftw_free(psi_y);
    fftw_free(psi_z);

    /* Destroy the FFTW plans */
    fftw_destroy_plan(r2c_x);
    fftw_destroy_plan(r2c_y);
    fftw_destroy_plan(r2c_z);
    fftw_destroy_plan(c2r_x);
    fftw_destroy_plan(c2r_y);
    fftw_destroy_plan(c2r_z);

    /* Clean up */
    cleanParams(&pars);

    sucmsg("test_poisson:\t SUCCESS");
}
