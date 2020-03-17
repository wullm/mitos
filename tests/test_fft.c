#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../include/dexm.h"

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
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *comp = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate a Gaussian random field */
    generate_complex_grf(fbox, N, boxlen);
    fft_apply_kernel(fbox, fbox, N, boxlen, sigma_func);

    /* Copy the complex array into a secondary memory to compare later */
    memcpy(comp, fbox, N*N*(N/2+1)*sizeof(fftw_complex));

    /* Transform to real configuration space */
    fft_execute(c2r);
	fft_normalize_c2r(box,N,boxlen);

    /* Export the Gaussian random field for testing purposes */
    const char box_fname[] = "test.box";
    write_doubles_as_floats(box_fname, box, N*N*N);
    printf("Example Gaussian random field exported to %s.\n", box_fname);

    /* FFT back */
    fft_execute(r2c);
    fft_normalize_r2c(fbox,N,boxlen);

    /* We should get back the original array, so let's verify. */
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                int idx = row_major_half(x, y, z, N);

                assert(fabs(fbox[idx] - comp[idx]) < 1e-5);
                // assert(fabs(fbox[idx][1] - comp[idx][1]) < 1e-5);
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
