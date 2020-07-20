#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "../include/dexm.h"

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
}

// static inline double sampleNorm() {
//     double sum = -6;
//     for (int i=0; i<12; i++) {
//         sum += (double) rand() / RAND_MAX;
//     }
//     return sum;
// }

int main() {
    /* Read parameters */
    const char fname[] = "test_cosmology.ini";
    struct params pars;
    struct units us;

    readParams(&pars, fname);
    readUnits(&us, fname);

    // /* Seed the random number generator */
    // srand(pars.Seed);

    /* Seed the xorshift random number generator */
    struct xoshiro256ss_state seed;
    seed.s[0] = pars.Seed;
    seed.s[1] = pars.Seed + 3;
    seed.s[2] = pars.Seed + 1;
    seed.s[3] = pars.Seed + 7;

    /* Test Gaussian random number generator */
    const int N = 1000000;
    double *x = malloc(N * sizeof(double));

    /* Generate the random numbers */
    for (int i=0; i<N; i++) {
        x[i] = sampleNorm(&seed);
    }

    /* Check sample mean */
    double xsum = 0;
    for (int i=0; i<N; i++) {
        xsum += x[i];
    }

    const double sample_mean = xsum/N;

    /* Check sample variance */
    double ssum = 0;
    for (int i=0; i<N; i++) {
        ssum += (x[i] - sample_mean)*(x[i] - sample_mean);
    }

    const double sample_sdev = sqrt(ssum / (N - 1.0));

    /* Conclusion */
    printf("sample_mean:\t %e\n", sample_mean);
    printf("sample_sdev:\t %e\n", sample_sdev);

    assert(fabs(sample_mean) < 1e-2);
    assert(fabs(sample_sdev - 1.0) < 1e-2);

    /* Get rid of the random numbers */
    free(x);

    /* Clean up */
    cleanParams(&pars);

    sucmsg("test_random:\t SUCCESS");
}
