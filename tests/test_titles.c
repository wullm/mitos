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

    /* Verify that we have the expected data */
    assert(trs.nrow == 616);
    assert(trs.n_functions == 26);

    int Nf = trs.n_functions;

    /* Verify some column titles */
    assert(strcmp(trs.titles[2], "d_cdm") == 0);
    assert(strcmp(trs.titles[24], "t_ncdm[2]") == 0);
    assert(strcmp(trs.titles[25], "t_tot") == 0);

    /* Assert that we can find titles */
    int d_cdm_id = find_title(trs.titles, "d_cdm", Nf);
    int d_ncdm_2_id = find_title(trs.titles, "t_ncdm[2]", Nf);
    int t_tot_id = find_title(trs.titles, "t_tot", Nf);

    assert(d_cdm_id == 2);
    assert(d_ncdm_2_id == 24);
    assert(t_tot_id == 25);

    /* Clean up */
    cleanTransfers(&trs);
    cleanParams(&pars);

    sucmsg("test_titles:\t SUCCESS");
}
