/*******************************************************************************
 * This file is part of Mitos.
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

#include <math.h>
#include <complex.h>
#include "../include/perturb_spline.h"
#include "../include/titles.h"
#include "../include/firebolt_interface.h"

/* Firebolt structures */
struct multipoles firebolt_mL; //multipoles in standard Legendre basis
struct multipoles firebolt_mmono; //multipoles in monomial basis
struct multipoles firebolt_mgauge; //Legendre multipole gauge transformations
struct grids firebolt_grs;

struct firebolt_helper {
    const struct perturb_spline *spline;
    int h_prime_index;
    int eta_prime_index;
    int delta_shift_index;
    int theta_shift_index;
};

struct firebolt_helper helper;

/* Redshift as a function of the logarithm of conformal time */
double redshift_func(double log_tau) {
    return perturbRedshiftAtLogTau(helper.spline, log_tau);
}

double h_prime_func(double k, double log_tau) {
    return perturbSplineInterp0(helper.spline, k, log_tau, helper.h_prime_index);
}

double eta_prime_func(double k, double log_tau) {
    return perturbSplineInterp0(helper.spline, k, log_tau, helper.eta_prime_index);
}

double delta_shift_func(double k, double log_tau) {
    return perturbSplineInterp0(helper.spline, k, log_tau, helper.delta_shift_index);
}

double theta_shift_func(double k, double log_tau) {
    return perturbSplineInterp0(helper.spline, k, log_tau, helper.theta_shift_index);
}

int initFirebolt(const struct params *pars, const struct cosmology *cosmo,
                 const struct units *us, const struct perturb_data *ptdat,
                 const struct perturb_spline *spline,
                 struct firebolt_interface *firebolt, const fftw_complex *grf,
                 double M_nu_eV, double T_nu_eV) {

    /* When is the simulation supposed to start? */
    double a_begin = 1.0 / (cosmo->z_ini + 1.0);
    double tau_begin_sim = exp(cosmo->log_tau_ini);

    printf("Running Firebolt.\n");

    /* Find indices corresponding to specific functions */
    helper.h_prime_index = findTitle(ptdat->titles, "h_prime", ptdat->n_functions);
    helper.eta_prime_index = findTitle(ptdat->titles, "eta_prime", ptdat->n_functions);
    helper.delta_shift_index = findTitle(ptdat->titles, "delta_shift_Nb_m", ptdat->n_functions);
    helper.theta_shift_index = findTitle(ptdat->titles, "t_cdm", ptdat->n_functions);
    helper.spline = spline;

    /* Make sure that all the necessary functions are there */
    if (helper.h_prime_index < 0 || helper.eta_prime_index < 0) {
        printf("ERROR: h'(k,tau) or eta'(k,tau) perturbation vectors missing.\n");
        return 1;
    }
    if (helper.delta_shift_index < 0 || helper.theta_shift_index < 0) {
        printf("ERROR: N-body gauge shift perturbation vectors missing.\n");
        return 1;
    }

    /* Dimensions of the gaussian random field */
    int N = pars->SmallGridSize;
    double boxlen = pars->BoxLen;

    /* Determine the maximum and minimum wavenumbers */
    double dk = 2*M_PI/boxlen;
    double k_max = sqrt(3)*dk*N/2;
    double k_min = dk;

    /* Ensure a safe error margin */
    k_max *= 1.5;
    k_min /= 1.5;

    /* The system to solve */
    int first_index = 0; //the earliest time is at the start
    double tau_ini = exp(ptdat->log_tau[first_index]);
    double tau_fin = tau_begin_sim; //integrate up to the beginning of the sim
    double a_fin = a_begin; //integrate up to the beginning of the sim

    /* Physical constants */
    const double M = M_nu_eV / T_nu_eV;
    const double c_vel = us->SpeedOfLight;

    /* Size of the problem */
    int l_max = pars->MaxMultipole;
    int l_max_convert = pars->MaxMultipoleConvert;
    int k_size = pars->NumberWavenumbers;
    double q_min = pars->MinMomentum;
    double q_max = pars->MaxMomentum;
    int q_steps = pars->NumberMomentumBins;
    double tol = pars->FireboltTolerance;
    short verbose = pars->FireboltVerbose;

    /* Store the momentum range in the renderer struct for later use */
    firebolt->q_size = q_steps;
    firebolt->log_q_min = log(q_min);
    firebolt->log_q_max = log(q_max);

    if (verbose) {
        printf("\n");
        printf("[k_min, k_max, k_size] = [%f, %f, %d]\n", k_min, k_max, k_size);
        printf("[l_max, q_steps, q_max, tol] = [%d, %d, %.1f, %.3e]\n", l_max, q_steps, q_max, tol);
        printf("[l_max_convert] = %d\n", l_max_convert);
        printf("\n");

        printf("The initial time is %e (z=%e)\n", tau_ini, redshift_func(log(tau_ini)));
        printf("The final time is %e (z=%e)\n", tau_fin, redshift_func(log(tau_fin)));
        printf("Speed of light c = %e\n", c_vel);
        printf("Neutrino mass M_nu = %f (%f eV)\n", M, M_nu_eV);
    }

    /* Initialize the multipoles */
    initMultipoles(&firebolt_mL, k_size, q_steps, l_max, q_min, q_max, k_min, k_max);

    /* Also initialize the multipoles in monomial basis (with much lower l_max) */
    initMultipoles(&firebolt_mmono, k_size, q_steps, l_max_convert+1, q_min, q_max, k_min, k_max);

    /* Initialize gauge transforms (only Psi_0 and Psi_1 are gauge dependent) */
    int l_size_gauge = 2;
    initMultipoles(&firebolt_mgauge, k_size, q_steps, l_size_gauge, q_min, q_max, k_min, k_max);

    /* Calculate the multipoles in Legendre basis */
    evolveMultipoles(&firebolt_mL, tau_ini, tau_fin, tol, M, c_vel, redshift_func, h_prime_func, eta_prime_func, verbose);

    /* Compute the gauge transforms in a separate struct (only Psi_0, Psi_1) */
    convertMultipoleGauge_Nb(&firebolt_mgauge, log(tau_fin), a_fin, M, c_vel, delta_shift_func, theta_shift_func);

    /* Convert from Legendre basis to monomial basis */
    convertMultipoleBasis_L2m(&firebolt_mL, &firebolt_mmono, l_max_convert);

    /* Convert the gauge transformations to monomial base and add it on top */
    convertMultipoleBasis_L2m(&firebolt_mgauge, &firebolt_mmono, 1);

    if (verbose) {
        printf("Done with integrating. Processing the moments.\n");
    }

    /* Initialize the multipole interpolation splines */
    initMultipoleInterp(&firebolt_mmono);

    /* Generate grids with the monomial multipoles */
    initGrids(N, boxlen, &firebolt_mmono, &firebolt_grs);

    /* Store reference to the grids */
    firebolt->grids_ref = &firebolt_grs;

    generateGrids(&firebolt_mmono, grf, &firebolt_grs);

    return 0;
}

int cleanFirebolt(void) {

    /* Clean up the Firebolt structures */
    cleanMultipoles(&firebolt_mmono);
    cleanMultipoles(&firebolt_mL);
    cleanMultipoles(&firebolt_mgauge);
    cleanMultipoleInterp();
    cleanGrids(&firebolt_grs);

    return 0;
}
