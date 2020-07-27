#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../include/mitos.h"

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
}

const double n_s = 0.97;
const double A_s = 2.215e-9;
const double k_pivot = 0.05;

static inline double sigma_func(double k) {
    if (k == 0) return 0;

    return sqrt(A_s * pow(k/k_pivot, n_s)) * kernel_transfer_function(k);
}

static inline void compute(struct kernel *the_kernel) {
    double k = the_kernel->k;
    double kern = sigma_func(k);
    the_kernel->kern = kern;
}

int main() {
    /* Read parameters */
    const char fname[] = "test_cosmology.ini";
    struct params pars;
    struct units us;
    struct cosmology cosmo;
    struct transfer trs;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, fname);

    /* Check if we can read the transfer function data file */
    const char *transfer_fname = pars.TransferFunctionsFile;
    FILE *f = fopen(transfer_fname, "r");
    assert(f != NULL);
    fclose(f);

    /* Read out the transfer functions */
    readTransfers(&pars, &us, &cosmo, &trs);

    /* Initialize the interpolation splines */
    tr_interp_init(&trs);

    /* Verify that we have the expected data */
    assert(trs.nrow == 616);
    assert(trs.n_functions == 26);

    /* Verify some column titles */
    assert(strcmp(trs.titles[2], "d_cdm") == 0);
    assert(strcmp(trs.titles[24], "t_ncdm[2]") == 0);
    assert(strcmp(trs.titles[25], "t_tot") == 0);

    /* Find the cdm density tranfer function */
    int id_d_cdm = find_title(trs.titles, "d_cdm", trs.n_functions);

    /* Switch the spline function */
    tr_interp_switch_func(&trs, id_d_cdm);

    printf("Hieron %d %f\n", id_d_cdm, kernel_transfer_function(1));

    /* Seed the random number generator */
    srand(pars.Seed);

    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    /* Create 3D arrays */
    double *box = (double*) fftw_malloc(N*N*N*sizeof(double));
    fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Create FFT plans */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);

    /* Generate a Gaussian random field */
    generate_complex_grf(fbox, N, boxlen);
    fft_apply_kernel(fbox, fbox, N, boxlen, compute);

    /* Transform to real configuraiton space */
    fft_execute(c2r);
	fft_normalize_c2r(box,N,boxlen);

    /* Export the Gaussian random field for testing purposes */
    const char box_fname[] = "test_trr.box";
    write_doubles_as_floats(box_fname, box, N*N*N);
    printf("Example Gaussian random field exported to %s.\n", box_fname);

    int bins = 50;
    double *k_in_bins = malloc(bins * sizeof(double));
    double *power_in_bins = malloc(bins * sizeof(double));
    int *obs_in_bins = calloc(bins, sizeof(int));

    /* Transform back to momentum space */
    fft_execute(r2c);
	fft_normalize_r2c(fbox,N,boxlen);

    calc_cross_powerspec(N, boxlen, fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

    /* Check that it is right */
    printf("\n");
    printf("Example power spectrum:\n");
    printf("k\t P_measured(k)\t P_input(k)\t observations\n");
    for (int i=0; i<bins; i++) {
        if (obs_in_bins[i] == 0) continue; //skip empty bins

        /* The power we observe */
        double k = k_in_bins[i];
        double Pk = power_in_bins[i];

        /* The power we expected */
        double sigma = sigma_func(k);
        double expected_Pk = sigma * sigma;

        /* The error (switch to log errors in the exponential tail) */
        double err, tol;
        if (expected_Pk < 1e-50) {
            err = 0.0; //no point checking this far
            tol = 0.0;
        } else if (expected_Pk < 1e-3) {
            err = fabs(log(Pk) - log(expected_Pk))/log(expected_Pk);
            tol = 0.1;
        } else {
            err = fabs(Pk - expected_Pk) / expected_Pk;
            tol = 0.7;
        }

        assert(err <= tol);

        printf("%f\t %e\t %e\t %d\n", k_in_bins[i], power_in_bins[i], expected_Pk, obs_in_bins[i]);
    }

    /* Free the memory */
    free(k_in_bins);
    free(power_in_bins);
    free(obs_in_bins);

    /* Free the remaining memory */
    fftw_free(box);
    fftw_free(fbox);

    /* Destroy the FFTW plans */
    tr_interp_free(&trs);
    fftw_destroy_plan(c2r);
    fftw_destroy_plan(r2c);

    /* Clean up */
    cleanParams(&pars);

    sucmsg("test_transfer_render:\t SUCCESS");
}
