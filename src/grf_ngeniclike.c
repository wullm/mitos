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

#include "../include/grf_ngeniclike.h"
#include "../include/primordial.h"
#include "../include/mitos.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <mpi.h>

int generate_ngeniclike_grf(fftw_complex * fbox, int N, int NX, int X0,
			                long int block_size, double boxlen, long int seed) {


    /* Convert Mitos variables to Ngenic variables */
    double Box = boxlen;
    long int Seed = seed;
    long int Nmesh = N;
    long int Nsample = N;
    int Local_nx = NX;
    int Local_x_start = X0;
    int SphereMode = 0;
    double PI = M_PI;
    fftw_complex *Cdata = fbox;

    gsl_rng *random_generator;
    int i, j, k, ii, jj, kk, axes;
    double fac, vel_prefac;
    double kvec[3], kmag, kmag2, p_of_k;
    double delta, phase, ampl, hubble_a;
    unsigned int *seedtable;

    /* Unused variables */
    (void) hubble_a;
	(void) vel_prefac;
	(void) kk;

    /* We do not backscale */
    double Dplus = 1.0;

    /* We need a different normalization factor than NGenic */
    // fac = pow (2 * PI / Box, 1.5);
    fac = pow(Box,1.5) / sqrt(2);

	fac *= sqrt(2); //to account for the different method of generating Gaussians

    /* GSL random number generator */
    random_generator = gsl_rng_alloc (gsl_rng_ranlxd1);
    gsl_rng_set (random_generator, Seed);

    if (!(seedtable = malloc (Nmesh * Nmesh * sizeof (unsigned int))))
    return 4;

    for (i = 0; i < Nmesh / 2; i++) {
        for (j = 0; j < i; j++)
        	seedtable[i * Nmesh + j] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i + 1; j++)
            seedtable[j * Nmesh + i] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i; j++)
	       seedtable[(Nmesh - 1 - i) * Nmesh + j] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i + 1; j++)
	       seedtable[(Nmesh - 1 - j) * Nmesh + i] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i; j++)
	       seedtable[i * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i + 1; j++)
	       seedtable[j * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i; j++)
	       seedtable[(Nmesh - 1 - i) * Nmesh + (Nmesh - 1 - j)] = 0x7fffffff * gsl_rng_uniform (random_generator);
        for (j = 0; j < i + 1; j++)
	       seedtable[(Nmesh - 1 - j) * Nmesh + (Nmesh - 1 - i)] = 0x7fffffff * gsl_rng_uniform (random_generator);
    }

    for (axes = 0; axes < 3; axes++) {

        /* first, clean the array */
        for (i = 0; i < Local_nx; i++) {
            for (j = 0; j < Nmesh; j++) {
                for (k = 0; k <= Nmesh / 2; k++) {
                    Cdata[(i * Nmesh + j) * (Nmesh / 2 + 1) + k] = 0;
                }
            }
        }

        for (i = 0; i < Nmesh; i++) {
	        ii = Nmesh - i;
	        if (ii == Nmesh) ii = 0;
	        if ((i >= Local_x_start && i < (Local_x_start + Local_nx)) ||
	            (ii >= Local_x_start && ii < (Local_x_start + Local_nx))) {

	            for (j = 0; j < Nmesh; j++) {
		            gsl_rng_set (random_generator, seedtable[i * Nmesh + j]);

    		    for (k = 0; k < Nmesh / 2; k++) {
    		        phase = gsl_rng_uniform (random_generator) * 2 * PI;
    		        do
    			    ampl = gsl_rng_uniform (random_generator);
    		        while (ampl == 0);

    		        if (i == Nmesh / 2 || j == Nmesh / 2 || k == Nmesh / 2)
    			    continue;
    		        if (i == 0 && j == 0 && k == 0)
    			    continue;

    		        if (i < Nmesh / 2) kvec[0] = i * 2 * PI / Box;
    		        else kvec[0] = -(Nmesh - i) * 2 * PI / Box;

    		        if (j < Nmesh / 2) kvec[1] = j * 2 * PI / Box;
    		        else kvec[1] = -(Nmesh - j) * 2 * PI / Box;

    		        if (k < Nmesh / 2) kvec[2] = k * 2 * PI / Box;
    		        else kvec[2] = -(Nmesh - k) * 2 * PI / Box;

    		        kmag2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2];
    		        kmag = sqrt (kmag2);

    		        if (SphereMode == 1) {
    			        if (kmag * Box / (2 * PI) > Nsample / 2)	/* select a sphere in k-space */
    			        continue;
    			    } else {
    			        if (fabs (kvec[0]) * Box / (2 * PI) > Nsample / 2)
    			        continue;
    			        if (fabs (kvec[1]) * Box / (2 * PI) > Nsample / 2)
    			        continue;
    			        if (fabs (kvec[2]) * Box / (2 * PI) > Nsample / 2)
    			        continue;
    			    }

    		        // p_of_k = PowerSpec (kmag);
                    p_of_k = 1.0; //power spectrum is applied later
					p_of_k *= -log (ampl);
    		        delta = fac * sqrt (p_of_k) / Dplus;	/* we do not scale back */

    		        if (k > 0) {
    			    if (i >= Local_x_start && i < (Local_x_start + Local_nx)) {
    			        double re = delta * sin (phase);
    			        double im = delta * cos (phase);

    			        Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k] = re + I * im;
    			    }
              } else { /* k=0 plane needs special treatment */
    		      if (i == 0) {
        	          if (j >= Nmesh / 2) continue;
        			  else {
        				  if (i >= Local_x_start && i < (Local_x_start + Local_nx)) {
        				      jj = Nmesh - j;	/* note: j!=0 surely holds at this point */

        				      double re = delta * sin (phase);
        				      double im = delta * cos (phase);

        				      Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k] = re + I * im;

        				      re = delta * sin (phase);
        				      im = -delta * cos (phase);
        				      Cdata[((i - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k] = re + I * im;
        				    }
        				}
                      } else { /* here comes i!=0 : conjugate can be on other processor! */
        		          if (i >= Nmesh / 2) continue;
        		          else {
        				    ii = Nmesh - i;
        				    if (ii == Nmesh) ii = 0;
        				    jj = Nmesh - j;
        				    if (jj == Nmesh) jj = 0;

                				    if (i >= Local_x_start && i < (Local_x_start + Local_nx)) {
                				        double re = delta * sin (phase);
                				        double im = delta * cos (phase);
                				        Cdata[((i - Local_x_start) * Nmesh + j) * (Nmesh / 2 + 1) + k] = re + I * im;
                				      }

                				    if (ii >= Local_x_start && ii < (Local_x_start + Local_nx)) {
                				        double re = delta * sin (phase);
                				        double im = -delta * cos (phase);
                				        Cdata[((ii - Local_x_start) * Nmesh + jj) * (Nmesh / 2 + 1) + k] = re + I * im;
                			        }
                                 }
        		             }
    			         }
    		         }
    		     }
	        }
	    }
    }

    return 0;
}
