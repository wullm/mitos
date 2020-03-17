#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <fftw3.h>

#include "../include/dexm.h"

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
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
    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;

    /* Create 3D arrays */
    double *box = (double*) fftw_malloc(N*N*N*sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *comp = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* Ignore the DC mode */
                if (k > 0) {
                    fbox[row_major_half(x,y,z,N)][0] = sampleNorm() * exp(-k*k/0.02) * sqrt(boxvol/2.0);
                    fbox[row_major_half(x,y,z,N)][1] = sampleNorm() * exp(-k*k/0.02) * sqrt(boxvol/2.0);
                } else {
                    fbox[row_major_half(x,y,z,N)][0] = 0;
                    fbox[row_major_half(x,y,z,N)][1] = 0;
                }
            }
        }
    }

    /* Enforce hermiticity: f(k) = f*(-k) */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z+=N/2) { //loops over k=0, N/2 only
                int invx = (x > 0) ? N - x : 0;
                int invy = (y > 0) ? N - y : 0;
                int invz = (z > 0) ? N - z : 0;
                fbox[row_major_half(x,y,z,N)][0] = fbox[row_major_half(invx,invy,invz,N)][0];
                fbox[row_major_half(x,y,z,N)][1] = -fbox[row_major_half(invx,invy,invz,N)][1];
            }
        }
    }

    /* Copy the complex array into a secondary memory to compare later */
    memcpy(comp, fbox, N*N*(N/2+1)*sizeof(fftw_complex));

    /* Transform to momentum space */
    fft_execute(c2r);
	fft_normalize_c2r(box,N,boxlen);

    // write_doubles_as_floats("box.box", box, N*N*N);

    /* FFT back */
    fft_execute(r2c);
    fft_normalize_r2c(fbox,N,boxlen);

    /* We should get back the original array, so let's verify. */
    int count  = 0;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                int idx = row_major_half(x, y, z, N);

                assert(fabs(fbox[idx][0] - comp[idx][0]) < 1e-5);
                assert(fabs(fbox[idx][1] - comp[idx][1]) < 1e-5);
            }
        }
    }

    /* Free it */
    fftw_free(box);
    fftw_free(fbox);
    fftw_free(comp);

    fftw_destroy_plan(r2c);
    fftw_destroy_plan(c2r);

    sucmsg("test_fft... SUCCESS");
}
