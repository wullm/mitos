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
    struct particle_type *types = NULL;

    readParams(&pars, fname);
    readUnits(&us, fname);
    readTypes(&pars, &types, fname);

    assert(pars.MaxParticleTypes == 3);
    assert(pars.NumParticleTypes == 3);

    for (int i=0; i<pars.NumParticleTypes; i++) {
        struct particle_type tp = types[i];
        printf("Found particle type\t %s\n", tp.Identifier);
    }


    /* Ensure that we find what we expected */
    struct particle_type cdm = types[0];
    assert(strcmp(cdm.Identifier, "cdm") == 0);
    assert(strcmp(cdm.ExportName, "PartType1") == 0);
    assert(cdm.Omega == 0.20);
    assert(cdm.CubeRootNumber == 64);
    assert(cdm.TotalNumber == 262144);
    assert(cdm.Chunks == 1);

    struct particle_type baryon = types[1];
    assert(strcmp(baryon.Identifier, "baryon") == 0);
    assert(strcmp(baryon.ExportName, "PartType2") == 0);
    assert(baryon.Mass == 12.5);
    assert(baryon.CubeRootNumber == 64);
    assert(baryon.TotalNumber == 262144);
    assert(baryon.Chunks == 1);

    struct particle_type neutrino = types[2];
    assert(strcmp(neutrino.Identifier, "neutrino") == 0);
    assert(strcmp(neutrino.ExportName, "PartType6") == 0);
    assert(neutrino.Omega == 0.01);
    assert(neutrino.CubeRootNumber == 64);
    assert(neutrino.TotalNumber == 262144);
    assert(neutrino.Chunks == 1);

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);


    sucmsg("test_types:\t SUCCESS");
}
