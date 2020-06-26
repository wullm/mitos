#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "../include/dexm.h"

static inline void sucmsg(const char *msg) {
    printf("%s%s%s\n\n", TXT_GREEN, msg, TXT_RESET);
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



    /* Assert that the interpolation works for each function */
    for (int id_func=0; id_func<trs.n_functions; id_func++) {
        /* Switch the spline function */
        tr_interp_switch_func(&trs, id_func);

        /* Check at the known values */
        for (int i=0; i<trs.nrow; i++) {
            double k = trs.k[i];
            double f = trs.functions[id_func][i];
            double f_interp = kernel_transfer_function(k);
            assert(fabs(f-f_interp)/f_interp < 1e-5);
        }

        /* Check inbetween the known values */
        for (int i=0; i<trs.nrow-1; i++) {
            double k1 = trs.k[i];
            double k2 = trs.k[i+1];
            double k_mid = 0.5*(k2 + k1);

            double f1 = trs.functions[id_func][i];
            double f2 = trs.functions[id_func][i+1];
            double f_mid = 0.5*(f1 + f2);

            double f_interp = kernel_transfer_function(k_mid);
            assert(fabs(f_mid-f_interp)/f_interp < 1e-2); //within 1% seems reasonable
        }
    }

    /* Clean up */
    tr_interp_free(&trs);
    cleanTransfers(&trs);
    cleanParams(&pars);

    sucmsg("test_transfer_interp:\t SUCCESS");
}
