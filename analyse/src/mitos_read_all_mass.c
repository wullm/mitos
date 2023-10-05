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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <assert.h>
#include <math.h>

#include "../../include/mitos.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Read options */
    const char *fname = argv[1];
    message(rank, "The parameter file is %s\n", fname);

    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct cosmology cosmo;
    struct perturb_data ptdat;
    struct perturb_params ptpars;
    struct perturb_spline spline;
    
    /* Read parameter file */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);
    readTypes(&pars, &types, fname);
    
    /* Option to override the input filename by specifying command line option */
    if (argc > 2) {
        const char *input_filename = argv[2];
        strcpy(pars.InputFilename, input_filename);
    }
    message(rank, "Reading simulation snapshot from: \"%s\".\n", pars.InputFilename);

    /* Option to disable output writing */
    int disable_write = 0;
    if (argc > 3) {
        disable_write = 1;
        message(rank, "Not writing output. Only computing power spectrum.\n");
    }

    /* Be verbose about the use of lossy output filters */
    if (pars.LossyScaleDigits > 0) {
        message(rank, "Using lossy output filter (keeping %d decimals).\n",
                pars.LossyScaleDigits);
    }
    
    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat, pars.PerturbFile);
    readPerturbParams(&pars, &us, &ptpars, pars.PerturbFile);

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Retrieve the physical density at z = 0 */

    /* The indices of the density transfer functions */
    int index = findTitle(ptdat.titles, pars.CrossSpectrumDensity1, ptdat.n_functions);
    int index_cdm = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
    int index_b = findTitle(ptdat.titles, "d_b", ptdat.n_functions);

    /* Determine the starting conformal time */
    cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);

    message(rank, "Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
    message(rank, "\n");

    /* Find the present-day background densities */
    int today_index = ptdat.tau_size - 1; // today corresponds to the last index
    double Omega_ncdm_z0 = ptdat.Omega[ptdat.tau_size * index + today_index];
    double Omega_ncdm_z = perturbDensityAtLogTau(&spline, cosmo.log_tau_ini, index);
    double Omega_cdm_z0 = ptdat.Omega[ptdat.tau_size * index_cdm + today_index];
    double Omega_cdm_z = perturbDensityAtLogTau(&spline, cosmo.log_tau_ini, index_cdm);
    double Omega_b_z0 = ptdat.Omega[ptdat.tau_size * index_b + today_index];
    double scaled_Omega_ncdm = Omega_ncdm_z / Omega_cdm_z * Omega_cdm_z0;
    double scaled_Omega_tot = scaled_Omega_ncdm + Omega_cdm_z0 + Omega_b_z0;
    
    double density_tot = rho_crit * scaled_Omega_tot;
    double density_nu_bg = rho_crit * scaled_Omega_ncdm;

    /* Critical density */
    double H_unit = 0.1022012156719; //100 km/s/Mpc in 1/Gyr
    double H = H_unit * ptpars.h;
    double G_newt = 4.49233855e-05; //Mpc^3/(1e10 M_sol)/Gyr^2 (my Mpc,Gyr,M_sol)
    double rho_crit = 3. * H * H / (8. * M_PI * G_newt);

    message(rank, "%d %d %g\n", index, index_cdm, Omega_z);
    message(rank, "Omega_ncdm = %g\n", Omega_ncdm_z0);
    message(rank, "Omega_tot = %g\n", Omega_tot_z0);
    message(rank, "Omega_cb = %g\n", Omega_cb_z0);
    message(rank, "density_cdm = %g\n", rho_crit * Omega_cdm_z0);
    message(rank, "density_b = %g\n", rho_crit * Omega_b_z0);
    message(rank, "density_tot = %g\n", density_tot);
    message(rank, "density_ncdm = %g\n", density_nu_bg);

    /* Open the file */
    // hid_t h_file = openFile_MPI(MPI_COMM_WORLD, pars.InputFilename);
    hid_t h_file = H5Fopen(pars.InputFilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    /* Open the Header group */
    hid_t h_grp = H5Gopen(h_file, "Header", H5P_DEFAULT);

    /* Read the physical dimensions of the box */
    double boxlen[3];
    hid_t h_attr = H5Aopen(h_grp, "BoxSize", H5P_DEFAULT);
    hid_t h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, boxlen);
    H5Aclose(h_attr);
    assert(h_err >= 0);

    boxlen[1] = boxlen[2] = boxlen[0];
    message(rank, "Reading particle type '%s'.\n", pars.ImportName);
    message(rank, "BoxSize is %g\n", boxlen[0]);

    /* Read the numbers of particles of each type */
    hsize_t numer_of_types;
    h_attr = H5Aopen(h_grp, "NumPart_Total", H5P_DEFAULT);
    hid_t h_atspace = H5Aget_space(h_attr);
    H5Sget_simple_extent_dims(h_atspace, &numer_of_types, NULL);
    H5Sclose(h_atspace);
    H5Aclose(h_attr);

    /* Close the Header group again */
    H5Gclose(h_grp);

    /* Check if the Cosmology group exists */
    hid_t h_status = H5Eset_auto1(NULL, NULL);  //turn off error printing
    h_status = H5Gget_objinfo(h_file, "/Cosmology", 0, NULL);

    /* If the group exists. */
    if (h_status == 0) {
        /* Open the Cosmology group */
        h_grp = H5Gopen(h_file, "Cosmology", H5P_DEFAULT);

        /* Read the redshift attribute */
        double redshift;
        h_attr = H5Aopen(h_grp, "Redshift", H5P_DEFAULT);
        h_err = H5Aread(h_attr, H5T_NATIVE_DOUBLE, &redshift);
        H5Aclose(h_attr);
        assert(h_err >= 0);

        message(rank, "The redshift was %f\n\n", redshift);

        /* Close the Cosmology group */
        H5Gclose(h_grp);
    }

    /* Try to open the desired import group */

    /* The size of the density grid that we will create */
    const int N = pars.GridSize;

    /* Allocate grids */
    double *box = fftw_alloc_real(N * N * N);
    
    double total_mass = 0;

    char typestrings[5][200] = {"PartType1", "PartType6", "PartType0", "PartType4", "PartType5"};

    for (int typenum = 0; typenum < 5; typenum++) {
        message(rank, "\nWorking on %s\n", typestrings[typenum]);

        //if (typenum == 1) continue;
        
        /* Open the corresponding group */
        h_grp = H5Gopen(h_file, typestrings[typenum], H5P_DEFAULT);

        /* Open the coordinates dataset */
        hid_t h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

        /* Find the dataspace (in the file) */
        hid_t h_space = H5Dget_space (h_dat);

        /* Get the dimensions of this dataspace */
        hsize_t dims[2];
        H5Sget_simple_extent_dims(h_space, dims, NULL);

        /* How many particles do we want per slab? */
        hid_t Npart = dims[0];
        hid_t max_slab_size = pars.SlabSize;
        int slabs = Npart/max_slab_size;
        hid_t counter = 0;

        /* Close the data and memory spaces */
        H5Sclose(h_space);

        /* Close the dataset */
        H5Dclose(h_dat);

        int slab_counter = 0;

        message(rank, "\n");

        for (int k=rank; k<slabs+1; k+=MPI_Rank_Count) {
            /* All slabs have the same number of particles, except possibly the last */
            hid_t slab_size = fmin(Npart - k * max_slab_size, max_slab_size);
            counter += slab_size; //the number of particles read

            /* Define the hyperslab */
            hsize_t slab_dims[2], start[2]; //for 3-vectors
            hsize_t slab_dims_one[1], start_one[1]; //for scalars

            /* Slab dimensions for 3-vectors */
            slab_dims[0] = slab_size;
            slab_dims[1] = 3; //(x,y,z)
            start[0] = k * max_slab_size;
            start[1] = 0; //start with x

            /* Slab dimensions for scalars */
            slab_dims_one[0] = slab_size;
            start_one[0] = k * max_slab_size;

            /* Open the coordinates dataset */
            h_dat = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);

            /* Find the dataspace (in the file) */
            h_space = H5Dget_space (h_dat);

            /* Select the hyperslab */
            hid_t status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
                                               NULL, slab_dims, NULL);
            assert(status >= 0);

            /* Create a memory space */
            hid_t h_mems = H5Screate_simple(2, slab_dims, NULL);

            /* Create the data array */
            double data[slab_size][3];

            status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                             data);

            /* Close the memory space */
            H5Sclose(h_mems);

            /* Close the data and memory spaces */
            H5Sclose(h_space);

            /* Close the dataset */
            H5Dclose(h_dat);


            /* Open the masses dataset (using DynamicalMasses for the BHs) */
            if (typenum == 4) {
                //h_dat = H5Dopen(h_grp, "SubgridMasses", H5P_DEFAULT);
                h_dat = H5Dopen(h_grp, "DynamicalMasses", H5P_DEFAULT);
            } else {
                h_dat = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
            }

            /* Find the dataspace (in the file) */
            h_space = H5Dget_space (h_dat);

            /* Select the hyperslab */
            status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                                slab_dims_one, NULL);

            /* Create a memory space */
            h_mems = H5Screate_simple(1, slab_dims_one, NULL);

            /* Create the data array */
            double mass_data[slab_size];

            status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                             mass_data);

            /* Close the memory space */
            H5Sclose(h_mems);

            /* Close the data and memory spaces */
            H5Sclose(h_space);

            /* Close the dataset */
            H5Dclose(h_dat);


            // /* Open the velocities dataset */
            // h_dat = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
            //
            // /* Find the dataspace (in the file) */
            // h_space = H5Dget_space (h_dat);
            //
            // /* Select the hyperslab */
            // status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start,
            //                              NULL, slab_dims, NULL);
            // assert(status >= 0);
            //
            // /* Create a memory space */
            // h_mems = H5Screate_simple(2, slab_dims, NULL);
            //
            // /* Create the data array */
            // double velocities_data[slab_size][3];
            //
            // status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
            //                  velocities_data);
            //
            // /* Close the memory space */
            // H5Sclose(h_mems);
            //
            // /* Close the data and memory spaces */
            // H5Sclose(h_space);
            //
            // /* Close the dataset */
            // H5Dclose(h_dat);
            
            /* Create the data array */
            double weight_data[slab_size];
            
            if (typenum == 1) {
                /* Open the weighs dataset */
                h_dat = H5Dopen(h_grp, "Weights", H5P_DEFAULT);

                /* Find the dataspace (in the file) */
                h_space = H5Dget_space (h_dat);

                /* Select the hyperslab */
                status = H5Sselect_hyperslab(h_space, H5S_SELECT_SET, start_one, NULL,
                                                    slab_dims_one, NULL);

                /* Create a memory space */
                h_mems = H5Screate_simple(1, slab_dims_one, NULL);

                status = H5Dread(h_dat, H5T_NATIVE_DOUBLE, h_mems, h_space, H5P_DEFAULT,
                                 weight_data);

                /* Close the memory space */
                H5Sclose(h_mems);

                /* Close the data and memory spaces */
                H5Sclose(h_space);

                /* Close the dataset */
                H5Dclose(h_dat);
            }

            double grid_cell_vol = boxlen[0]*boxlen[1]*boxlen[2] / (N*N*N);

            /* Assign the particles to the grid with CIC */
            for (int l=0; l<slab_size; l++) {
                double x = fwrap(data[l][0], boxlen[0]);
                double y = fwrap(data[l][1], boxlen[0]);
                double z = fwrap(data[l][2], boxlen[0]);

                double X = x / (boxlen[0]/N);
                double Y = y / (boxlen[1]/N);
                double Z = z / (boxlen[2]/N);

                double M = mass_data[l];

                total_mass += M;
                
                if (typenum == 1) {
                    double w = weight_data[l];
                    M *= w;
                }
                
                //total_mass += M;

                int iX = (int) floor(X);
                int iY = (int) floor(Y);
                int iZ = (int) floor(Z);

    	        /* Displacements from grid corner */
                double dx = X - iX;
                double dy = Y - iY;
                double dz = Z - iZ;
                double tx = 1.0 - dx;
                double ty = 1.0 - dy;
                double tz = 1.0 - dz;
            
                //double val = M / grid_cell_vol;
                double val = M;

                /* Deposit the mass over the nearest 8 cells */
                box[row_major(iX, iY, iZ, N)] += val * tx * ty * tz;
                box[row_major(iX+1, iY, iZ, N)] += val * dx * ty * tz;
                box[row_major(iX, iY+1, iZ, N)] += val * tx * dy * tz;
                box[row_major(iX, iY, iZ+1, N)] += val * tx * ty * dz;
                box[row_major(iX+1, iY+1, iZ, N)] += val * dx * dy * tz;
                box[row_major(iX+1, iY, iZ+1, N)] += val * dx * ty * dz;
                box[row_major(iX, iY+1, iZ+1, N)] += val * tx * dy * dz;
                box[row_major(iX+1, iY+1, iZ+1, N)] += val * dx * dy * dz;

            }

            printf("(%03d,%03d) Read %ld particles\n", rank, k, slab_size);
            slab_counter++;
        }

        /* Close the group again */
        H5Gclose(h_grp);
    }

    /* Reduce the grid */
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, box, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(box, box, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* Reduce the total mass */
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(&total_mass, &total_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }


    if (rank == 0) {

        message(rank, "Total mass: %g\n", total_mass);

        /* The average density */
        double avg_density = total_mass / (boxlen[0]*boxlen[1]*boxlen[2]);
        message(rank, "Average density %g\n", avg_density);

        /* Reset the density */
        //avg_density = Omega_tot_z0 * rho_crit;
        message(rank, "Density expected from perturb file: %g\n", (density_tot - density_nu_bg));
        
        /* The cb density */
        //double avg_density_cb = Omega_cb_z0 * rho_crit;
        
        /* Turn the density field into an overdensity field */
        for (int x=0; x<N; x++) {
            for (int y=0; y<N; y++) {
                for (int z=0; z<N; z++) {
                    int id = row_major(x, y, z, N);
                    box[id] = (box[id] + density_nu_bg) / density_tot - 1.;
                }
            }
        }

        /* Find a particle type with a matching ExportName */
        struct particle_type *tp;
        char found = 0;
        for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
            struct particle_type *ptype = types + pti;
            const char *ExportName = ptype->ExportName;

            if (strcmp(ExportName, pars.ImportName) == 0) {
                tp = ptype;
                found = 1;
            }
        }

        if (disable_write == 0) {
            char box_fname[40];
            if (!found) {
                sprintf(box_fname, "density_%s.hdf5", pars.ImportName);
            } else {
                sprintf(box_fname, "density_%s.hdf5", tp->Identifier);
            }
            writeFieldFileCompressed(box, N, boxlen[0], box_fname, pars.LossyScaleDigits);
            message(rank, "Density grid exported to %s.\n", box_fname);
        }
    }

    if (rank == 0) {
        int bins = pars.PowerSpectrumBins;
        double *k_in_bins = malloc(bins * sizeof(double));
        double *power_in_bins = malloc(bins * sizeof(double));
        int *obs_in_bins = calloc(bins, sizeof(int));

        /* Transform to momentum space */
        fftw_complex *fbox = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));
        fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, box, fbox, FFTW_ESTIMATE);
        fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fbox, box, FFTW_ESTIMATE);
        fft_execute(r2c);
    	fft_normalize_r2c(fbox,N,boxlen[0]);

        /* Undo the CIC window function */
        undoCICWindow(fbox, N, boxlen[0]);

        calc_cross_powerspec(N, boxlen[0], fbox, fbox, bins, k_in_bins, power_in_bins, obs_in_bins);

        /* Check that it is right */
        printf("\n");
        printf("Example power spectrum:\n");
        printf("k P_measured(k) observations\n");
        for (int i=0; i<bins; i++) {
            if (obs_in_bins[i] <= 1) continue; //skip (virtually) empty bins

            /* The power we observe */
            double k = k_in_bins[i];
            double Pk = power_in_bins[i];
            int obs = obs_in_bins[i];

            printf("%f %e %d\n", k, Pk, obs);
        }

        fftw_destroy_plan(r2c);
        fftw_destroy_plan(c2r);
        fftw_free(fbox);

        printf("\n");
    }
    fftw_free(box);

    /* Close the HDF5 file */
    H5Fclose(h_file);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanTypes(&pars, &types);
    cleanParams(&pars);
}
