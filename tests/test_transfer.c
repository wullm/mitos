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
    struct transfer trs;

    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Check if we can read the transfer function data file */
    const char *transfer_fname = pars.TransferFunctionsFile;
    FILE *f = fopen(transfer_fname, "r");
    assert(f != NULL);
    fclose(f);

    /* Read out the transfer functions */
    readTransfers(&pars, &trs);

    /* Verify that we have the expected data */
    assert(trs.nrow == 616);
    assert(trs.ncol == 27);

    /* Verify some column titles */
    assert(strcmp(trs.titles[0], "k") == 0);
    assert(strcmp(trs.titles[3], "d_cdm") == 0);
    assert(strcmp(trs.titles[25], "t_ncdm[2]") == 0);
    assert(strcmp(trs.titles[26], "t_tot") == 0);

    /* Verify some data values */
    assert(abs(trs.k[0] - 1.062036233582e-05)/trs.k[0] < 1e-5);
    assert(abs(trs.functions[2][0] - -2.987327466470e-05)/trs.functions[2][0] < 1e-5);
    assert(abs(trs.k[615] - 1.110201190377e+01)/trs.k[615] < 1e-5);
    assert(abs(trs.functions[2][615] - -2.543613922206e+03)/trs.functions[2][615] < 1e-5);

    /* Clean up */
    cleanTransfers(&trs);

    sucmsg("test_transfer:\t SUCCESS");
}
