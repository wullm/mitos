#include <stdio.h>
#include <stdlib.h>

#define sizearray(a)  (sizeof(a) / sizeof((a)[0]))

#include "dexm.h"

const char inifile[] = "default.ini";

int main() {
    printf("Fine,\n");

    float f = ini_getf("Box", "Seed", 1.0, inifile);
    printf("%f\n", f);
}
