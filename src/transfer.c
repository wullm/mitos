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
        /* Skip the first character, which should be '#' */
        char title[50];
        int j = 0, read = 1, bytes;

        /* Read the column titles */
        while(sscanf(line + read, "%s%n", title, &bytes) > 0) {
            titles[j] = malloc(strlen(title) + 1);
            strcpy(titles[j], title);

            read += bytes;
            j++;
        }
    } else if (format == CLASS) {
        /* Skip the first character, which should be '#' */
        char title[50];
        int j = 0, col, read = 1, bytes;

        /* We need to adjust for a space in the first column title,
         * which looks like "1:k (h/Mpc)".
         */

        /* Read the first part of the first column "1:k" */
        sscanf(line + read, "%d:%s%n", &col, title, &bytes);
        read += bytes;

        /* Store the first column title (just "k") */
        titles[0] = malloc(strlen(title) + 1);
        strcpy(titles[0], title);
        j++;

        /* Next read the second bit of the first column "(h/Mpc)" */
        sscanf(line + read, "%s%n", title, &bytes);
        read += bytes;

        /* Now read the remaining column titles, which contain no spaces */
        while(sscanf(line + read, "%d:%s%n", &col, title, &bytes) > 0) {
            titles[j] = malloc(strlen(title) + 1);
            strcpy(titles[j], title);

            read += bytes;
            j++;
        }
    }
}

int readTransfers(const struct params *pars, struct transfer *trs) {
    const char *fname = pars->TransferFunctionsFile;
    const char *formatString = pars->TransferFunctionsFormat;
    enum transfer_format format;

    /* Parse the expected format of the transfer funtion file */
    if (strcmp(formatString, "Plain") == 0) {
        format = Plain;
    } else if (strcmp(formatString, "CLASS") == 0) {
        format = CLASS;
    } else {
        printf("ERROR: Unknown transfer functions format.");
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

    return 0;
}


int cleanTransfers(struct transfer *trs) {
    free(trs->titles);
    free(trs->functions);
    free(trs->k);

    return 0;
}
