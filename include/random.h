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

#ifndef RANDOM_H
#define RANDOM_H

#define SEARCH_TABLE_LENGTH 1000
#define NUMERICAL_CDF_SAMPLES 1000

#define THERMAL_MIN_MOMENTUM 1e-10 //don't use exactly zero
#define THERMAL_MAX_MOMENTUM 15.0
#define FERMION_TYPE "fermion"
#define BOSON_TYPE "boson"

/* We allow for user-defined pdf's */
typedef double (*pdf)(double x, void *params);

struct interval {
    int id;
    double l, r;            // endpoints left and right
    double Fl, Fr;          // cdf evaluations at endpoints
    double a0, a1, a2, a3;  // cubic Hermite coefficients
    double error;           // error at midpoint
    int nid;                // the next interval
};

struct sampler {
    /* The normalization of the pdf */
    double norm;

    /* The endpoints */
    double xl, xr;

    /* Pointer to the probability density function */
    pdf f;

    /* Array of optional parameters passed to the pdf */
    void *params;

    /* The intervals used in the interpolation */
    struct interval *intervals;

    /* The number of intervals */
    int intervalNum;

    /* Length of the indexed search table */
    int I_max;

    /* The indexed search table */
    double *index;
};

/* Compare intervals by the value of the CDF at the left endpoint */
static inline int compareByLeft(const void *a, const void *b) {
    struct interval *ia = (struct interval*) a;
    struct interval *ib = (struct interval*) b;
    return ia->Fl >= ib->Fl;
}

double sampleNorm();
double fd_pdf(double x, void *params);
double be_pdf(double x, void *params);
double numericalCDF(double xl, double xr, int samples, pdf f, void *params);
int initSampler(struct sampler *s, pdf f, double xl, double xr, void *params);
int splitInterval(struct sampler *s, int current_interval_id);
int cleanSampler(struct sampler *s);
double samplerCustom(struct sampler *s);

#endif
