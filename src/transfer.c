/*******************************************************************************
 * This file is part of DEXM.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <ctype.h> //isdigit
#include <math.h>
#include "../include/transfer.h"

static inline char is_comment_line(char *line) {
    /* Lines starting with '#' or that don't begin with a number */
    float tmp;
    return line[0] == '#' || (sscanf(line, "%e", &tmp) <= 0);
}

void countRowsCols(FILE *f, int *nrow, int *ncol, int *leading_comment_lines) {
    char line[2000];

    /* Local counters */
    int rows = 0;
    int cols = 0;
    int comments = 0;

    /* Count the number of rows with data */
    while (fgets(line, sizeof(line), f)) {
        if (is_comment_line(line)) continue; /* skip comments */
        rows++;
    }

    /* Go back to the start */
    rewind(f);

    /* Count the number of leading comment lines */
    while (fgets(line, sizeof(line), f) && is_comment_line(line)) {
        comments++;
    }

    /* On the first non-comment line, count the number of columns */
    int read = 0, bytes;
    float tmp;
    while(sscanf(line + read, "%e%n", &tmp, &bytes) > 0) {
        read += bytes;
        cols += 1;
    }

    /* Update the counters */
    *nrow = rows;
    *ncol = cols;
    *leading_comment_lines = comments;
}


void readTitles(char *line, enum transfer_format format, char **titles) {
    if (format == Plain) {
        /* The Plain format is "# k name name name " */
        /* We want to read out the column names */

        /* Skip the first character, which should be '#' */
        char title[50];
        int j = 0, read = 0, bytes;

        /* Read the column titles & count the # of bytes read */
        while(sscanf(line + read, "%s%n", title, &bytes) > 0) {
            titles[j] = malloc(strlen(title) + 1);
            strcpy(titles[j], title);

            read += bytes;
            j++;
        }
    } else if (format == CLASS) {
        /* The CLASS format is "# 1:k    2:name     3:name    4:name ...." */
        /* We want to read out the column names */

        /* Skip the first character, which should be '#' */
        char string[50];
        int j = 0, read = 1, bytes;

        /* Read string until encountering a colon & count the # of bytes read */
        while(sscanf(line + read, "%[^:]%n", string, &bytes) > 0) {
            /* Parse the title from the full string */
            int string_read = 0, string_bytes; //second counter within string
            char part[50], title[50];
            strcpy(title, "");
            /* Read words (separated by spaces) until encountering a number */
            while (sscanf(string + string_read, "%s%n", part, &string_bytes) > 0
                   && !isdigit(part[0])) {
                strcat(title, part);
                strcat(title, " ");
                string_read += string_bytes;
            }

            /* If we found a non-empty title, store it */
            if (strlen(title) > 0) {
                title[strlen(title)-1] = '\0'; /* delete trailing whitespace */
                titles[j] = malloc(strlen(title) + 1);
                strcpy(titles[j], title);
                j++;
            }

            read += bytes + 1;
        }
    }
}

int readTransfers(const struct params *pars, const struct units *us,
                  const struct cosmology *cosmo, struct transfer *trs) {
    const char *fname = pars->TransferFunctionsFile;
    const char *formatString = pars->TransferFunctionsFormat;
    enum transfer_format format;

    /* Parse the expected format of the transfer funtion file */
    if (strcmp(formatString, "Plain") == 0) {
        format = Plain;
    } else if (strcmp(formatString, "CLASS") == 0) {
        format = CLASS;
    } else {
        printf("ERROR: Unknown transfer functions format.\n");
        return 1;
    }

    /* Open the data file */
    FILE *f = fopen(fname, "r");
    char line[2000];

    /* Determine the size of the table */
    int nrow;
    int ncol;
    int leading_comment_lines;
    countRowsCols(f, &nrow, &ncol, &leading_comment_lines);

    trs->nrow = nrow;
    trs->ncol = ncol;

    /* Move to the last leading comment line */
    rewind(f);
    for (int i=0; i<leading_comment_lines; i++) {
        fgets(line, sizeof(line), f);
    }

    /* Allocate memory for pointers to the column title strings */
    trs->titles = malloc(ncol * sizeof(char*));

    /* Read out the column titles */
    readTitles(line, format, trs->titles);

    /* Change the first title to "k" if that was not already the case */
    if (strcmp(trs->titles[0], "k") != 0) {
        trs->titles[0] = realloc(trs->titles[0], 2);
        strcpy(trs->titles[0], "k");
    }

    /* Allocate memory for the transfer function data */
    trs->k = malloc(nrow * sizeof(float));
    trs->functions = malloc(ncol * sizeof(float*));
    for (int i=0; i<ncol; i++) {
        trs->functions[i] = malloc(nrow * sizeof(float));
    }

    /* Read out the data into the arrays */
    int row = 0;
    float number = 0;
    while (fgets(line, sizeof(line), f)) {
        if (is_comment_line(line)) continue; /* skip comments */
        int read = 0, bytes;
        int col = 0;
        while(sscanf(line + read, "%e%n", &number, &bytes) > 0) {
            if (col == 0) {
                trs->k[row] = number;
            } else {
                trs->functions[col-1][row] = number;
            }
            read += bytes;
            col++;
        }
        row++;
    }

    fclose(f);

    /* Internally, the h-exponent of the wavenumbers is 0
     * (i.e. 1/Mpc not h/Mpc) and the k-exponent of the transfer
     * functions is -2 as it is for CAMB/CMBFAST/Eisenstein-Hu.
     */

    double factor;

    /* First, adjust the h-exponent (i.e. convert from h/Mpc to 1/Mpc) */
    factor = pow(cosmo->h, us->Transfer_hExponent);

    /* Next, convert to internal length units */
    factor /= us->TransferUnitLengthMetres;
    factor *= us->UnitLengthMetres;

    /* Update the wavenumbers */
    for (int i=0; i<nrow; i++) {
        trs->k[i] *= factor;
    }

    /* Finally, adjust the k-exponent of the transfer functions */
    for (int i=0; i<ncol; i++) {
        for (int j=0; j<nrow; j++) {
            trs->functions[i][j] /= pow(trs->k[j], us->Transfer_kExponent + 2);
            trs->functions[i][j] *= us->Transfer_Sign; //an overall sign flip
        }
    }

    /* A useful warning if the format does not match the exponents */
    if (format == CLASS && us->Transfer_Sign != -1) {
        printf("Warning: TransferFunctions:Sign != -1 as expected for CLASS.\n");
    }
    if (format == CLASS && us->Transfer_hExponent != 1) {
        printf("Warning: TransferFunctions:hExponent != 1 as expected for CLASS.\n");
    }
    if (format == CLASS && us->Transfer_kExponent != 0) {
        printf("Warning: TransferFunctions:kExponent != 0 as expected for CLASS.\n");
    }

    return 0;
}


int cleanTransfers(struct transfer *trs) {
    free(trs->titles);
    free(trs->functions);
    free(trs->k);

    return 0;
}
