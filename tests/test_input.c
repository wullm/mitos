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
    /* Test reading parameters */
    const char fname[] = "test_cosmology.ini";
    struct params pars;

    readParams(&pars, fname);

    // assert(pars.Seed == 100);
    assert(pars.BoxLen == 256.0);
    assert(pars.GridSize == 64);
    assert(strcmp(pars.Name, "Test Simulation") == 0);
    assert(strcmp(pars.OutputDirectory, "../tests") == 0);
    // assert(strcmp(pars.TransferFunctionsFile, "../transfer_functions/example_class_output_tk.dat") == 0);
    // assert(strcmp(pars.TransferFunctionsFormat, "CLASS") == 0);

    /* Test reading units */
    struct units us;
    readUnits(&us, fname);

    assert(us.UnitLengthMetres == 3.086e22);
    assert(us.UnitTimeSeconds == 3.154e16);
    assert(us.UnitMassKilogram == 1.989e40);


    sucmsg("test_input:\t SUCCESS");
}
