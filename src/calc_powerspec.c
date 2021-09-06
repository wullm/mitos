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

#include <hdf5.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#include "../include/calc_powerspec.h"
#include "../include/fft.h"

static inline double sinc(double x) {
    return (x == 0) ? 1 : sin(x)/x;
}

void calc_cross_powerspec(int N, double boxlen, const fftw_complex *box1,
                          const fftw_complex *box2, int bins, double *k_in_bins,
                          double *power_in_bins, int *obs_in_bins) {

    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        k_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k==0) continue; //skip the DC mode

                /* Compute the bin */
                const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                const int bin = floor((bins - 1) * u);
                const long long int id = row_major_half(x, y, z, N);

                assert(bin >= 0 && bin < bins);

                /* Compute the power <X,Y> with X,Y complex */
                double a1 = creal(box1[id]), a2 = creal(box2[id]);
                double b1 = cimag(box1[id]), b2 = cimag(box2[id]);
                double Power = a1*a2 + b1*b2;

                /* All except the z=0 and the z=N/2 planes count double */
                int multiplicity = (z==0 || z==N/2) ? 1 : 2;

                /* Add to the tables */
                k_in_bins[bin] += multiplicity * k;
				power_in_bins[bin] += multiplicity * Power;
				obs_in_bins[bin] += multiplicity;
            }
        }
    }

    /* Divide to obtain averages */
	for (int i=0; i<bins; i++) {
		k_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= boxvol;
	}
}

void calc_cross_powerspec_float(int N, double boxlen, const fftwf_complex *box1,
                                const fftwf_complex *box2, int bins, double *k_in_bins,
                                double *power_in_bins, int *obs_in_bins) {

    const double boxvol = boxlen*boxlen*boxlen;
    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        k_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double kx,ky,kz,k;
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k==0) continue; //skip the DC mode

                /* Compute the bin */
                const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                const int bin = floor((bins - 1) * u);
                const long long int id = row_major_half(x, y, z, N);

                assert(bin >= 0 && bin < bins);

                /* Compute the power <X,Y> with X,Y complex */
                double a1 = creal(box1[id]), a2 = creal(box2[id]);
                double b1 = cimag(box1[id]), b2 = cimag(box2[id]);
                double Power = a1*a2 + b1*b2;

                /* All except the z=0 and the z=N/2 planes count double */
                int multiplicity = (z==0 || z==N/2) ? 1 : 2;

                /* Add to the tables */
                k_in_bins[bin] += multiplicity * k;
				power_in_bins[bin] += multiplicity * Power;
				obs_in_bins[bin] += multiplicity;
            }
        }
    }

    /* Divide to obtain averages */
	for (int i=0; i<bins; i++) {
		k_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= boxvol;
	}
}

void calc_cross_powerspec_2d(int N, double anglesize, const fftw_complex *box1,
                             const fftw_complex *box2, int bins, double *l_in_bins,
                             double *power_in_bins, int *obs_in_bins) {

    const double norm = pow(anglesize*M_PI/180/(N*N),2);
    const double dl = 360.0/anglesize;
    const double max_l = sqrt(3)*dl*N/2;
    const double min_l = dl;

    const double log_max_l = log(max_l);
    const double log_min_l = log(min_l);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        l_in_bins[i] = 0;
        power_in_bins[i] = 0;
        obs_in_bins[i] = 0;
    }

    /* Calculate the power spectrum */
    double lx,ly,l;
    for (int x=0; x<N; x++) {
        for (int y=0; y<=N/2; y++) {
            /* Calculate the wavevector */
            lx = (x > N/2) ? (x - N)*dl : x*dl;
            ly = (y > N/2) ? (y - N)*dl : y*dl;
            l = sqrt(lx*lx + ly*ly);

            if (l==0) continue; //skip the DC mode

            /* Compute the bin */
            const float u = (log(l) - log_min_l) / (log_max_l - log_min_l);
            const int bin = floor((bins - 1) * u);
            const int id = x*(N/2+1) + y;

            assert(bin >= 0 && bin < bins);

            /* Compute the power <X,Y> with X,Y complex */
            double a1 = creal(box1[id]), a2 = creal(box2[id]);
            double b1 = cimag(box1[id]), b2 = cimag(box2[id]);
            double Power = a1*a2 + b1*b2;

            /* All except the z=0 and the z=N/2 planes count double */
            int multiplicity = (y==0 || y==N/2) ? 1 : 2;

            /* Add to the tables */
            l_in_bins[bin] += multiplicity * l;
			power_in_bins[bin] += multiplicity * Power;
			obs_in_bins[bin] += multiplicity;
        }
    }

    /* Divide to obtain averages */
	for (int i=0; i<bins; i++) {
		l_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] /= obs_in_bins[i];
		power_in_bins[i] *= norm;
	}
}

/* Method for calculating the bispectrum B(k1, k2, k3) of an N^3 grid.
 *
 * In volume mode, the input box (fbox) is ignored and a unit grid is assumed.
 * This can be used to compute the bin volumes for normalizing the bispectrum.
 *
 * The type can be:  0 = based on a fixed angle theta between k1 and k2
 *                   1 = based on a fixed k3 value
 *
 * For type = 0 the value of k3 is ignored. For type = 1 the value of theta is
 * ignored.
 *
 * Examples:
 * For equilateral bispectrum, k1 = k2 = k3, use type = 0 and theta = 4 pi / 3.
 * For squeezed bispectrum, k1 = k2 >> k3, use type 1 and specify a small k3.
 */
void calc_bispectrum(int N, double boxlen, const fftw_complex *fbox,
                     int bins, int bins3, double *k1_in_bins,
                     double *k2_in_bins, double *k3_in_bins, double *bispectrum,
                     int volume_mode, int type, double theta, double k3) {

    const double dk = 2*M_PI/boxlen;
    const double max_k = sqrt(3)*dk*N/2;
    const double min_k = dk;

    const double log_max_k = log(max_k);
    const double log_min_k = log(min_k);

    /* Reset the bins */
    for (int i=0; i<bins; i++) {
        k1_in_bins[i] = 0;
        k2_in_bins[i] = 0;
        k3_in_bins[i] = 0;
        bispectrum[i] = 0;
    }

    /* NB: box2 is not currently needed as we assume k1 = k2. Nevertheless,
     * we leave the infrastructure in pace. */
    int second_box_required = 0;

    /* Allocate intermediate masked boxes */
    double *masked1 = (double*) malloc((long long)N*N*N*sizeof(double));
    double *masked2;
    double *masked3 = (double*) malloc((long long)N*N*N*sizeof(double));
    fftw_complex *fbox1 = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
    fftw_complex *fbox2;
    fftw_complex *fbox3 = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));

    if (second_box_required) {
        masked2 = (double*) malloc((long long)N*N*N*sizeof(double));
        fbox2 = (fftw_complex*) malloc((long long)N*N*(N/2+1)*sizeof(fftw_complex));
    }

    /* Mask the boxes and then Fourier transform to real space */
    printf("bin k1 k2 k3 k1_obs k2_obs k3_obs Bk\n");
    for (int i = 0; i < bins; i++) {

        /* Collect the mean value of each k value */
        double k1_mean = 0.;
        double k2_mean = 0.;
        double k3_mean = 0.;
        long long k1_obs = 0;
        long long k2_obs = 0;
        long long k3_obs = 0;

        /* We are assuming that k1 = k2 = k for now */
        int required_bin1 = i;
        int required_bin2 = i;
        int required_bin3;

        /* Determine the required k3 bin */
        if (type == 0) {
            /* Compute the corresponding k3 value if using a theta */
            /* NB: we may use a different number of bins for k3: bins != bins3 */
            double k1 = exp((i + 0.5) * (log_max_k - log_min_k) / (bins3 - 1) + log_min_k);
            double k2 = exp((i + 0.5) * (log_max_k - log_min_k) / (bins3 - 1) + log_min_k);
            double required_k3 = sqrt(k1 * k1 + k2 * k2 + 2 * k1 * k2 * cos(theta));
            float u3 = (log(required_k3) - log_min_k) / (log_max_k - log_min_k);
            required_bin3 = floor((bins3 - 1) * u3);
        } else if (type == 1) {
            /* Use a fixed value of k3 and its surrounding bin */
            double required_k3 = k3;
            float u3 = (log(required_k3) - log_min_k) / (log_max_k - log_min_k);
            required_bin3 = floor((bins3 - 1) * u3);
        } else {
            required_bin3 = 0;
            printf("ERROR: Unknown bispectrum type!");
        }

        /* Make masked grids based on the above binning */
        double kx,ky,kz,k;
        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<=N/2; z++) {
                    /* Calculate the wavevector */
                    fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                    /* The grid cell id */
                    const long long int id = row_major_half(x, y, z, N);

                    /* Skip the DC mode */
                    if (k==0) {
                        fbox1[id] = 0.;
                        fbox3[id] = 0.;
                        if (second_box_required) {
                            fbox2[id] = 0.;
                        }
                        continue;
                    }

                    /* Compute the bin corresponding to this wavenumber */
                    const float u = (log(k) - log_min_k) / (log_max_k - log_min_k);
                    const int bin = floor((bins - 1) * u);
                    assert(bin >= 0 && bin < bins);

                    /* Also compute bin3 (since possibly bins != bin3) */
                    const int bin3 = floor((bins3 - 1) * u);
                    assert(bin3 >= 0 && bin3 < bins3);

                    /* All except the z=0 and the z=N/2 planes count double */
                    int multiplicity = (z==0 || z==N/2) ? 1 : 2;

                    /* Mask the grid for k1 */
                    if (bin == required_bin1) {
                        fbox1[id] = (volume_mode) ? 1.0 : fbox[id];
                        k1_mean += multiplicity * k;
                        k1_obs += multiplicity;
                    } else {
                        fbox1[id] = 0.0;
                    }

                    /* Mask the grid for k3 */
                    if (bin3 == required_bin3) {
                        fbox3[id] = (volume_mode) ? 1.0 : fbox[id];
                        k3_mean += multiplicity * k;
                        k3_obs += multiplicity;
                    } else {
                        fbox3[id] = 0.0;
                    }

                    /* We can skip the second box if we assume k1 = k2 */
                    if (!second_box_required) continue;

                    /* Mask the grid for k2 */
                    if (bin == required_bin2) {
                        fbox2[id] = (volume_mode) ? 1.0 : fbox[id];
                        k2_mean += multiplicity * k;
                        k2_obs += multiplicity;
                    } else {
                        fbox1[id] = 0.0;
                    }
                }
            }
        }

        /* We don't separately mask the second box if we assume k1 = k2 */
        if (!second_box_required) {
            k2_mean = k1_mean;
            k2_obs = k1_obs;
        }

        /* Skip empty bins */
        if (k1_obs == 0 || k2_obs == 0 || k3_obs == 0) continue;

        /* Determine the mean wavenumbers */
        k1_mean /= k1_obs;
        k2_mean /= k2_obs;
        k3_mean /= k3_obs;

        /* Fourier the first box to real space */
        fftw_plan c2r1 = fftw_plan_dft_c2r_3d(N, N, N, fbox1, masked1, FFTW_ESTIMATE);
        fft_execute(c2r1);
        fft_normalize_c2r(masked1,N,boxlen);
        fftw_destroy_plan(c2r1);

        /* Fourier the third box to real space */
        fftw_plan c2r3 = fftw_plan_dft_c2r_3d(N, N, N, fbox3, masked3, FFTW_ESTIMATE);
        fft_execute(c2r3);
        fft_normalize_c2r(masked3,N,boxlen);
        fftw_destroy_plan(c2r3);

        if (second_box_required) {
            /* Fourier the second box to real space */
            fftw_plan c2r2 = fftw_plan_dft_c2r_3d(N, N, N, fbox2, masked2, FFTW_ESTIMATE);
            fft_execute(c2r2);
            fft_normalize_c2r(masked2,N,boxlen);
            fftw_destroy_plan(c2r2);
        }

        /* Compute the unnormalized bispectrum for this bin combination */
        double sum = 0.;
        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    const long long int id = row_major(x, y, z, N);

                    double delta_1 = masked1[id];
                    double delta_2;
                    double delta_3 = masked3[id];

                    /* We can use masked2 = masked1 if k1 = k2 */
                    if (second_box_required) {
                        delta_2 = masked2[id];
                    } else {
                        delta_2 = masked1[id];
                    }

                    sum += delta_1 * delta_2 * delta_3;
                }
            }
        }

        bispectrum[i] = sum;
        k1_in_bins[i] = k1_mean;
        k2_in_bins[i] = k2_mean;
        k3_in_bins[i] = k3_mean;

        printf("%d %g %g %g %lld %lld %lld %.10g\n", i, k1_mean, k2_mean, k3_mean, k1_obs, k2_obs, k3_obs, sum);
    }

    /* Free the masked boxes */
    free(masked1);
    free(fbox1);
    free(masked3);
    free(fbox3);
    if (second_box_required) {
        free(masked2);
        free(fbox2);
    }
}
