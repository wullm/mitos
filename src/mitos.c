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
#include <sys/time.h>
#include <assert.h>
#include <complex.h>

#include "../include/mitos.h"
#include "../include/grf_ngeniclike.h"

#define COMPILED_WITH_FIREBOLT 0
#define FIREBOLT_EXPLICIT_CHECKS 1000

#if(COMPILED_WITH_FIREBOLT)
#include "../include/firebolt_interface.h"
#endif

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
    if (rank == 0) {
        header(rank, "Mitos Initial Condition Generator");
        message(rank, "The parameter file is '%s'\n", fname);
    }

    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);

    /* Mitos structuress */
    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct export_group *export_groups = NULL;
    struct cosmology cosmo;
    struct perturb_data ptdat;
    struct perturb_data ptdat_alternative;
    struct perturb_spline spline;
    struct perturb_spline spline_alternative;
    struct perturb_params ptpars;

    #if(COMPILED_WITH_FIREBOLT)
    struct firebolt_interface firebolt;
    #endif

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);

    /* Store the MPI rank */
    pars.rank = rank;

    message(rank, "The output directory is '%s'.\n", pars.OutputDirectory);
    message(rank, "Creating initial conditions for '%s'.\n", pars.Name);

    /* Read out particle types from the parameter file */
    readTypes(&pars, &types, fname);

    /* Match particle types with export groups */
    fillExportGroups(&pars, &types, &export_groups);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat, pars.PerturbFile);
    readPerturbParams(&pars, &us, &ptpars, pars.PerturbFile);

    /* Did the user requested a second alternative perturbation data file? */
    if (strcmp(pars.SecondPerturbFile, "") != 0) {
        message(pars.rank, "Opening another perturbation data file.\n");
        readPerturb(&pars, &us, &ptdat_alternative, pars.SecondPerturbFile);

        /* Initialize the interpolation spline for the second data file */
        initPerturbSpline(&spline_alternative, DEFAULT_K_ACC_TABLE_SIZE,
                          &ptdat_alternative);
    }

    /* Do a sanity check */
    if (fabs(cosmo.h - ptpars.h) / cosmo.h > 1e-5) {
        catch_error(1, "ERROR: h from parameter file does not match perturbation file.\n");
    }

    /* Merge cdm & baryons into one set of transfer functions (replacing cdm) */
    if (pars.MergeDarkMatterBaryons) {
        header(rank, "Merging cdm & baryon transfer functions, replacing cdm.");

        /* The indices of the density transfer functions */
        int index_cdm = findTitle(ptdat.titles, "d_cdm", ptdat.n_functions);
        int index_b = findTitle(ptdat.titles, "d_b", ptdat.n_functions);

        /* Find the present-day background densities */
        int today_index = ptdat.tau_size - 1; // today corresponds to the last index
        double Omega_cdm = ptdat.Omega[ptdat.tau_size * index_cdm + today_index];
        double Omega_b = ptdat.Omega[ptdat.tau_size * index_b + today_index];

        /* Do a sanity check */
        assert(fabs(Omega_b - ptpars.Omega_b) / Omega_b < 1e-5);

        /* Use the present-day densities as weights */
        double weight_cdm = Omega_cdm / (Omega_cdm + Omega_b);
        double weight_b = Omega_b / (Omega_cdm + Omega_b);

        message(rank, "Using weights [w_cdm, w_b] = [%f, %f]\n", weight_cdm, weight_b);

        /* Merge the density & velocity transfer runctions, replacing cdm */
        mergeTransferFunctions(&ptdat, "d_cdm", "d_b", weight_cdm, weight_b);
        mergeTransferFunctions(&ptdat, "t_cdm", "t_b", weight_cdm, weight_b);
        /* Merge the background densities, replacing cdm */
        mergeBackgroundDensities(&ptdat, "d_cdm", "d_b", 1.0, 1.0); //replace with sum
    }

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + rank);

    /* Determine the starting conformal time */
    cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);
    /* Determine the conformal time at the source redshift (usually z_ini) */
    cosmo.log_tau_source = perturbLogTauAtRedshift(&spline, cosmo.z_source);

    /* Should we use the primary perturbation data or the alternative data
     * (if specified) for the growth factors? */
     struct perturb_spline *growth_factors_spline;
     if (pars.GrowthFactorsFromSecondFile) {
         growth_factors_spline = &spline_alternative;
         if (strcmp(pars.SecondPerturbFile, "") == 0) {
             catch_error(1, "Requested growth factors from second file without specifying second perturbation file.\n");
         }
     } else {
         growth_factors_spline = &spline;
     }

    /* Compute the growth factors and rates (!! careful with log_taus from the alterative file !!)*/
    const double log_tau_source = perturbLogTauAtRedshift(growth_factors_spline, cosmo.z_source);
    const double log_tau_ini = perturbLogTauAtRedshift(growth_factors_spline, cosmo.z_ini);
    const double a_source = 1.0 / (1.0 + cosmo.z_source);
    const double a_ini = 1.0 / (1.0 + cosmo.z_ini);
    const double D_source = perturbGrowthFactorAtLogTau(growth_factors_spline, log_tau_source);
    const double D_ini = perturbGrowthFactorAtLogTau(growth_factors_spline, log_tau_ini);
    const double f_source = perturbLogGrowthRateAtLogTau(growth_factors_spline, log_tau_source);
    const double f_ini = perturbLogGrowthRateAtLogTau(growth_factors_spline, log_tau_ini);
    const double H_source = perturbHubbleAtLogTau(growth_factors_spline, log_tau_source);
    const double H_ini = perturbHubbleAtLogTau(growth_factors_spline, log_tau_ini);

    /* Print some useful numbers */
    if (rank == 0) {
        header(rank, "Settings");
        printf("Random numbers\t\t [seed] = [%ld]\n", pars.Seed);
        printf("Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
        printf("Source time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_source, exp(cosmo.log_tau_source));
        printf("Primordial power\t [A_s, n_s, k_pivot] = [%.4e, %.4f, %.4f U_L]\n\n", cosmo.A_s, cosmo.n_s, cosmo.k_pivot);

        if (pars.GrowthFactorsFromSecondFile) {
            printf("Using growth factors from second file: '%s'.\n", pars.SecondPerturbFile);
        } else {
            printf("Using growth factors from file: '%s'.\n", pars.PerturbFile);
        }

        printf("Growth factors\t\t [D_ini, D_source, ratio] = [%e, %e, %e]\n", D_ini, D_source, D_ini / D_source);
        printf("Growth rates\t\t [f_ini, f_source, ratio] = [%e, %e, %e]\n", f_ini, f_source, f_ini / f_source);
        printf("Hubble rates\t\t [H_ini, H_source, ratio] = [%e, %e, %e]\n", H_ini, H_source, H_ini / H_source);

        header(rank, "Requested Particle Types");
        for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
            /* The current particle type */
            struct particle_type *ptype = types + pti;
            printf("Particle type '%s' (N^3 = %d^3).\n", ptype->Identifier, ptype->CubeRootNumber);
        }
    }

    /* Create or read a Gaussian random field */
    int N;
    double boxlen;
    struct distributed_grid grf;

    if (strcmp(pars.ReadGaussianFileName, "") == 0) {
        /* Create Gaussian random field */
        N = pars.GridSize;
        boxlen = pars.BoxLen;

        /* Allocate distributed memory arrays (one complex & one real) */
        alloc_local_grid(&grf, N, boxlen, MPI_COMM_WORLD);

        /* Generate a complex Hermitian Gaussian random field */
        header(rank, "Generating Primordial Fluctuations");
        generate_complex_grf(&grf, &seed);
        enforce_hermiticity(&grf);

        /* Apply the bare power spectrum, without any transfer functions */
        fft_apply_kernel_dg(&grf, &grf, kernel_power_no_transfer, &cosmo);

        /* Execute the Fourier transform and normalize */
        fft_c2r_dg(&grf);
    } else {
        /* Prepare to read the Gassian random field */
        header(rank, "Reading Primordial Fluctuations");
        message(rank, "Reading file '%s'.\n", pars.ReadGaussianFileName);

        /* Read the field dimensions */
        readFieldDimensions(&N, &boxlen, pars.ReadGaussianFileName);

        message(rank, "Read (N, BoxLen) = (%d, %.2f U_L)\n", N, boxlen);

        /* Allocate distributed memory arrays (one complex & one real) */
        alloc_local_grid(&grf, N, boxlen, MPI_COMM_WORLD);

        /* Read the real-space grid from the file */
        readFieldFile_dg(&grf, pars.ReadGaussianFileName);
    }

    /* Generate a filename */
    char grf_fname[DEFAULT_STRING_LENGTH];
    sprintf(grf_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");

    /* Export the real GRF */
    int err = writeFieldFile_dg(&grf, grf_fname);
    catch_error(err, "Error while writing '%s'.\n", fname);
    message(rank, "Pure Gaussian Random Field exported to '%s'.\n", grf_fname);

    /* Create smaller (zoomed out) copies of the Gaussian random field */
    for (int i=0; i<2; i++) {
        /* Size of the smaller grid */
        int M;

        /* Generate a filename */
        char small_fname[DEFAULT_STRING_LENGTH];

        /* We do this twice, once if the user requests a SmallGridSize and
         * another time if the user requests a FireboltGridSize. */
        if (i == 0) {
            M = pars.SmallGridSize;
            sprintf(small_fname, "%s/%s%s", pars.OutputDirectory,  GRID_NAME_GAUSSIAN_SMALL, ".hdf5");
        } else {
            M = pars.FireboltGridSize;
            sprintf(small_fname, "%s/%s%s", pars.OutputDirectory,  GRID_NAME_GAUSSIAN_FIREBOLT, ".hdf5");
        }

        if (M > 0) {
            /* Allocate memory for the smaller grid on each node */
            double *grf_small = fftw_alloc_real(M * M * M);

            /* Shrink (our local slice of) the larger grf grid */
            shrinkGrid_dg(grf_small, &grf, M, N);

            /* Add the contributions from all nodes and send it to the root node */
            if (rank == 0) {
                MPI_Reduce(MPI_IN_PLACE, grf_small, M * M * M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            } else {
                MPI_Reduce(grf_small, grf_small, M * M * M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }

            /* Export the assembled smaller copy on the root node */
            if (rank == 0) {
                writeFieldFile(grf_small, M, boxlen, small_fname);
                message(rank, "Smaller copy of the Gaussian Random Field exported to '%s'.\n", small_fname);
            }

            /* Free the small grid */
            fftw_free(grf_small);
        }
    }



    /* Go back to momentum space */
    fft_r2c_dg(&grf);

    /* Retrieve background densities from the perturbations data file */
    header(rank, "Fetching Background Densities");
    retrieveDensities(&pars, &cosmo, &types, &ptdat);
    retrieveMicroMasses(&pars, &cosmo, &types, &ptpars);

    /* Allocate a second grid to compute densities */
    struct distributed_grid grid;
    alloc_local_grid(&grid, N, boxlen, MPI_COMM_WORLD);

    /* Allocate a third grid to compute the potential */
    struct distributed_grid potential;
    alloc_local_grid(&potential, N, boxlen, MPI_COMM_WORLD);

    /* Allocate a fourth grid to compute derivatives */
    struct distributed_grid derivative;
    alloc_local_grid(&derivative, N, boxlen, MPI_COMM_WORLD);

    /* Sanity check */
    assert(grf.local_size == grid.local_size);
    assert(grf.local_size == derivative.local_size);
    assert(grf.X0 == grid.X0);
    assert(grf.X0 == derivative.X0);


    /* We calculate derivatives using FFT kernels */
    const kernel_func derivative_kernels[] = {kernel_dx, kernel_dy, kernel_dz};
    const char *letter[] = {"x_", "y_", "z_"};

    header(rank, "Computing Perturbation Grids");

    /* For each particle type, compute displacement & velocity grids */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;
        const char *density_title = ptype->TransferFunctionDensity;
        const char *velocity_title = ptype->TransferFunctionVelocity;

        /* Generate filenames for the grid exports */
        char density_filename[DEFAULT_STRING_LENGTH];
        char potential_filename[DEFAULT_STRING_LENGTH];
        char velocity_filename[DEFAULT_STRING_LENGTH];
        char velopot_filename[DEFAULT_STRING_LENGTH];
        char derivative_filename[DEFAULT_STRING_LENGTH];

        generateFieldFilename(&pars, density_filename, Identifier, GRID_NAME_DENSITY, "");
        generateFieldFilename(&pars, potential_filename, Identifier, GRID_NAME_POTENTIAL, "");
        generateFieldFilename(&pars, velocity_filename, Identifier, GRID_NAME_THETA, "");
        generateFieldFilename(&pars, velopot_filename, Identifier, GRID_NAME_THETA_POTENTIAL, "");

        /* Generate density field, compute the potential and its derivatives */
        if (strcmp("", density_title) != 0) {

            message(rank, "Computing density & displacement grids for '%s'.\n", Identifier);

            /* If we are backscaling, multiply by the growth factor ratio */
            double rescale_factor;
            if (cosmo.z_ini != cosmo.z_source) {
                rescale_factor = D_ini / D_source;
            } else {
                rescale_factor = 1.0;
            }

            /* Should we generate a density field or load it from the disk? */
            if (strcmp(ptype->InputFilenameDensity, "") == 0) {
              /* Generate density grid by applying the transfer function to the GRF */
              err = generatePerturbationGrid(&cosmo, &spline, &grf, &grid, density_title, density_filename, rescale_factor);
              catch_error(err, "Error while generating '%s'.", density_filename);
            } else {
               /* Load input density field */
               message(rank, "Loading density grid from '%s'.\n", ptype->InputFilenameDensity);
               err = readFieldFile_dg(&grid, ptype->InputFilenameDensity);
               catch_error(err, "Error while loading '%s'.", ptype->InputFilenameDensity);
            }

            /* Fourier transform the density grid */
            fft_r2c_dg(&grid);

            /* Should we solve the Monge-Ampere equation or approximate with Zel'dovich? */
            if (ptype->CyclesOfMongeAmpere > 0) {
                /* Solve the Monge Ampere equation */
                err = solveMongeAmpere(&potential, &grid, &derivative, ptype->CyclesOfMongeAmpere);
            } else if (ptype->Run2LPT > 0) {
                /* Solve for the 2LPT potential */
                err = solve2LPT(&potential, &grid, &derivative, 1.0, -3./7.);
            } else {
                /* Approximate the potential with the Zel'dovich approximation */
                fft_apply_kernel_dg(&potential, &grid, kernel_inv_poisson, NULL);
            }

            /* We now have the potential grid in momentum space */
            assert(potential.momentum_space == 1);

            /* Undo the TSC window function for later */
            struct Hermite_kern_params Hkp;
            Hkp.order = 3; //TSC
            Hkp.N = N;
            Hkp.boxlen = boxlen;

            /* Apply the kernel */
            fft_apply_kernel_dg(&potential, &potential, kernel_undo_Hermite_window, &Hkp);

            /* Compute three derivatives of the potential grid */
            for (int i=0; i<3; i++) {
                /* Apply the derivative kernel */
                fft_apply_kernel_dg(&derivative, &potential, derivative_kernels[i], NULL);

                /* Fourier transform to get the real derivative grid */
                fft_c2r_dg(&derivative);

                /* Generate the appropriate filename */
                generateFieldFilename(&pars, derivative_filename, Identifier, GRID_NAME_DISPLACEMENT, letter[i]);

                /* Export the derivative grid */
                writeFieldFile_dg(&derivative, derivative_filename);
            }

            /* Finally, Fourier transform the potential grid to configuration space */
            fft_c2r_dg(&potential);

            /* Export the potential grid */
            writeFieldFile_dg(&potential, potential_filename);
        }

        /* Generate flux density field, flux potential, and its derivatives */
        if (strcmp("", velocity_title) != 0) {

            message(rank, "Computing flux density & velocity grids for '%s'.\n", Identifier);

            /* If we are backscaling, multiply by the DaHf ratio */
            double rescale_factor;
            if (cosmo.z_ini != cosmo.z_source) {
                rescale_factor = (a_ini * f_ini * H_ini) / (a_source * f_source * H_source);
            } else {
                rescale_factor = 1.0;
            }

            /* Should we generate a flux density field or load it from the disk? */
            if (strcmp(ptype->InputFilenameVelocity, "") == 0) {
              /* Generate flux grid by applying the transfer function to the GRF */
              err = generatePerturbationGrid(&cosmo, &spline, &grf, &grid, velocity_title, velocity_filename, rescale_factor);
              catch_error(err, "Error while generating '%s'.", velocity_filename);
            } else {
               /* Load input flux density field */
               message(rank, "Loading flux density grid from '%s'.\n", ptype->InputFilenameVelocity);
               err = readFieldFile_dg(&grid, ptype->InputFilenameVelocity);
               catch_error(err, "Error while loading '%s'.", ptype->InputFilenameVelocity);
            }

            /* Fourier transform the flux density grid */
            fft_r2c_dg(&grid);

            if (ptype->Run2LPT > 0) {
                /* Solve for the 2LPT potential */
                err = solve2LPT(&potential, &grid, &derivative, -0.001147273637728, 3.19354920304769E-05);
            } else {
                /* Compute flux potential grid by applying the inverse Poisson kernel */
                fft_apply_kernel_dg(&potential, &grid, kernel_inv_poisson, NULL);
            }

            /* Undo the TSC window function for later */
            struct Hermite_kern_params Hkp;
            Hkp.order = 3; //TSC
            Hkp.N = N;
            Hkp.boxlen = boxlen;

            /* Apply the kernel */
            fft_apply_kernel_dg(&potential, &potential, kernel_undo_Hermite_window, &Hkp);

            /* Compute three derivatives of the flux potential grid */
            for (int i=0; i<3; i++) {
                /* Apply the derivative kernel */
                fft_apply_kernel_dg(&derivative, &potential, derivative_kernels[i], NULL);

                /* Fourier transform to get the real derivative grid */
                fft_c2r_dg(&derivative);

                /* Generate the appropriate filename */
                generateFieldFilename(&pars, derivative_filename, Identifier, GRID_NAME_VELOCITY, letter[i]);

                /* Export the derivative grid */
                writeFieldFile_dg(&derivative, derivative_filename);
            }

            /* Finally, Fourier transform the flux potential grid to configuration space */
            fft_c2r_dg(&potential);

            /* Export the flux potential grid */
            writeFieldFile_dg(&potential, velopot_filename);
        }
    }

    /* We are done with the GRF, density, and derivative grids */
    free_local_grid(&grid);
    free_local_grid(&potential);
    free_local_grid(&grf);
    free_local_grid(&derivative);

    // /* Compute SPT grids */
    // header(rank, "Computing SPT Corrections");
    // err = computePerturbedGrids(&pars, &us, &cosmo, types, GRID_NAME_DENSITY, GRID_NAME_THETA);
    // if (err > 0) exit(1);



    /* Create the beginning of a SWIFT parameter file */
    if (rank == 0) {
        header(rank, "Creating SWIFT Parameter File");
        char out_par_fname[DEFAULT_STRING_LENGTH];
        sprintf(out_par_fname, "%s/%s", pars.OutputDirectory, pars.SwiftParamFilename);
        printf("Creating output file '%s'.\n", out_par_fname);
        writeSwiftParameterFile(&pars, &cosmo, &us, &types, &ptpars, out_par_fname);
    }

    /* Name of the main output file containing the initial conditions */
    header(rank, "Initializing Output File");
    char out_fname[DEFAULT_STRING_LENGTH];
    sprintf(out_fname, "%s/%s", pars.OutputDirectory, pars.OutputFilename);
    message(rank, "Creating output file '%s'.\n", out_fname);

    if (rank == 0) {
        /* Create the output file */
        hid_t h_out_file = createFile(out_fname);

        /* Writing attributes into the Header & Cosmology groups */
        err = writeHeaderAttributes(&pars, &cosmo, &us, &types, h_out_file);
        if (err > 0) exit(1);

        /* Create an HDF5 Group for each ExportName */
        for (int i = 0; i < pars.NumExportGroups; i++) {
            /* The current export group */
            struct export_group *grp = export_groups + i;

            /* The total number of particles that are mapped to this ExportName */
            long long int partnum = grp->TotalNumber;

            /* The ExportName */
            const char *ExportName = grp->ExportName;

            /* The particle group in the output file */
            hid_t h_grp;

            /* Datsets */
            hid_t h_data;

            /* Vector dataspace (e.g. positions, velocities) */
            const hsize_t vrank = 2;
            const hsize_t vdims[2] = {partnum, 3};
            hid_t h_vspace = H5Screate_simple(vrank, vdims, NULL);

            /* Scalar dataspace (e.g. masses, particle ids) */
            const hsize_t srank = 1;
            const hsize_t sdims[1] = {partnum};
            hid_t h_sspace = H5Screate_simple(srank, sdims, NULL);

            /* Create the particle group in the output file */
            printf("Creating Group '%s' with %lld particles.\n", ExportName, partnum);
            h_grp = H5Gcreate(h_out_file, ExportName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            /* Coordinates (use vector space) */
            h_data = H5Dcreate(h_grp, "Coordinates", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Velocities (use vector space) */
            h_data = H5Dcreate(h_grp, "Velocities", H5T_NATIVE_DOUBLE, h_vspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Masses (use scalar space) */
            h_data = H5Dcreate(h_grp, "Masses", H5T_NATIVE_DOUBLE, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Particle IDs (use scalar space) */
            h_data = H5Dcreate(h_grp, "ParticleIDs", H5T_NATIVE_LLONG, h_sspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dclose(h_data);

            /* Close the group */
            H5Gclose(h_grp);
        }

        /* Close the file */
        H5Fclose(h_out_file);
    }

    /* Now open the file in parallel mode */
    hid_t h_out_file = openFile_MPI(MPI_COMM_WORLD, out_fname);

    /* For each user-defined particle type */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;

        char str[50];
        sprintf(str, "Generating Particle Type '%s'.", ptype->Identifier);
        header(rank, str);

        /* Skip empty particle types */
        if (ptype->TotalNumber <= 0) {
            printf("No particles requested.\n");
            continue;
        }

        /* ID of the first particle of this type */
        const long long int id_first_particle = ptype->FirstID;

        /* Random sampler used for thermal species */
        struct sampler thermal_sampler;

        /* For diagnostics, count how many draws are needed from the
         * thermal distribution */
        long long thermal_draws = 0;

        /* For diagnostics, estimate the correlation between the 0th order
         * phase space density perturbation Psi_0 & the configuration space
         * density perturbation. (Should be order 1.)  */
        double Psi_sum = 0.;
        double Psi2_sum = 0.;
        double d_sum = 0.;
        double d2_sum = 0.;
        double Psi_d_sum = 0.;

        /* For diagnostics, estimate the ratio of particles whose 1st order
         * phase space density perturbation Psi_1 is positive along the
         * graviational flow speed direction. (Should be order 1.) */
        long long int correctly_oriented = 0;
        long long int explicit_Psi_checks = 0;

        /* Microscopic mass in electronVolts */
        double M_eV;
        /* Convert the temperature to electronVolts */
        double T_eV;
        /* Chemical potential */
        double mu_eV;

        /* Small grid, potentially used for the Firebolt Boltzmann code */
        #if(COMPILED_WITH_FIREBOLT)
        fftw_complex *small_grf = NULL;
        #endif

        /* Initialize a random sampler if this particle type is thermal */
        if (strcmp(ptype->ThermalMotionType, "") != 0) {
            /* Check if the type of thermal motion is supported */
            pdf function;
            /* Domain of the probability function */
            double xl, xr;

            if (strcmp(ptype->ThermalMotionType, FERMION_TYPE) == 0) {
                function = fd_pdf;
                xl = THERMAL_MIN_MOMENTUM; //units of kb*T
                xr = THERMAL_MAX_MOMENTUM; //units of kb*T
            } else if (strcmp(ptype->ThermalMotionType, BOSON_TYPE) == 0) {
                function = be_pdf;
                xl = THERMAL_MIN_MOMENTUM; //units of kb*T
                xr = THERMAL_MAX_MOMENTUM; //units of kb*T
            } else {
                printf("ERROR: unsupported ThermalMotionType '%s'.\n", ptype->ThermalMotionType);
                exit(1);
            }

            /* Microscopic mass in electronVolts */
            M_eV = ptype->MicroscopicMass_eV;
            /* Convert the temperature to electronVolts */
            T_eV = ptype->MicroscopyTemperature * us.kBoltzmann / us.ElectronVolt;
            /* Chemical potential */
            mu_eV = 0;

            /* Rescale the domain */
            xl *= T_eV;
            xr *= T_eV;

            /* Initialize the sampler */
            double thermal_params[2] = {T_eV, mu_eV};

            err = initSampler(&thermal_sampler, function, xl, xr, thermal_params);
            if (err > 0) {
                printf("Error initializing the thermal motion sampler.\n");
                exit(1);
            }

            message(rank, "Thermal motion: %s with [M, T] = [%e eV, %e eV].\n",
                    ptype->ThermalMotionType, M_eV, T_eV);

            /* Beyond the zeroth order Fermi-Dirac distribution, we can use the
             * linear theory perturbation from a Boltzmann code. */

            /* Use the Firebolt Boltzmann code */
            #if(COMPILED_WITH_FIREBOLT)
            if (ptype->UseFirebolt) {

                /* Grid size used for the Firebolt Boltzmann code */
                int K;

                /* Load the correct Gaussian random field */
                char read_small_fname[DEFAULT_STRING_LENGTH];

                /* If the user specified a FireboltGridSize, use that grid */
                if (pars.FireboltGridSize > 0) {
                    K = pars.FireboltGridSize;
                    sprintf(read_small_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN_FIREBOLT, ".hdf5");
                }
                /* Otherwise, if the SmallGridSize was specified, use that */
                else if (pars.SmallGridSize > 0) {
                    K = pars.SmallGridSize;
                    sprintf(read_small_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN_SMALL, ".hdf5");
                }
                /* Otherwise, use the full Gaussian random field */
                else {
                    K = N;
                    sprintf(read_small_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");
                }

                /* Allocate small complex 3D array */
                small_grf = (fftw_complex*) malloc(K*K*(K/2+1)*sizeof(fftw_complex));

                /* Read the real Gaussian field from disk */
                int read_N;
                double read_boxlen;
                double *small_grid;
                err = readFieldFile(&small_grid, &read_N, &read_boxlen, read_small_fname);
                if (err > 0) {
                    printf("Error while loading the Gaussian random field for Firebolt.\n");
                    exit(1);
                }

                /* Check that the sizes match what we expect */
                if (read_N != K || read_boxlen != boxlen) {
                    printf("Incorrect field dimensions in file (%d, %f) != (%d, %f)\n", read_N, read_boxlen, N, boxlen);
                    exit(1);
                }

                /* Compute the Fourier transform */
                fftw_plan small_r2c = fftw_plan_dft_r2c_3d(K, K, K, small_grid, small_grf, FFTW_ESTIMATE);
                fft_execute(small_r2c);
                fft_normalize_r2c(small_grf, K, boxlen);

                /* Free the real box, because we only need the complex grid */
                free(small_grid);
                fftw_destroy_plan(small_r2c);

                /* Initialize the Firebolt Boltzmann code */
                initFirebolt(&pars, &cosmo, &us, &ptdat, &spline, &firebolt, small_grf, M_eV, T_eV);
            }
            #endif
        }

        /* The particle group in the output file */
        hid_t h_grp = H5Gopen(h_out_file, ptype->ExportName, H5P_DEFAULT);

        /* Datsets */
        hid_t h_data;

        /* Vector dataspace (e.g. positions, velocities) */
        const hsize_t vrank = 2;
        h_data = H5Dopen2(h_grp, "Velocities", H5P_DEFAULT);
        hid_t h_vspace = H5Dget_space(h_data);
        H5Dclose(h_data);

        /* Scalar dataspace (e.g. masses, particle ids) */
        const hsize_t srank = 1;
        h_data = H5Dopen2(h_grp, "ParticleIDs", H5P_DEFAULT);
        hid_t h_sspace = H5Dget_space(h_data);
        H5Dclose(h_data);

        /* We read the displacement & velocity fields as distributed grids.
         * Once again, the local slice will be of size NX * N * (N + 2),
         * where the last two rows are padding and contain no useful info. */

        /* The local slice runs from local_X0 <= X < local_X0 + local_NX */
        int local_X0 = grf.X0;
        int local_NX = grf.NX;

        /* The particles are also generated from a grid with dimension M^3 */
        int M = ptype->CubeRootNumber;

        /* Determine what particles belong to this slice */
        double fac = (double) M / N;
        int X_min = ceil(local_X0 * fac);
        int X_max = ceil((local_X0 + local_NX) * fac);
        int MX = X_max - X_min;

        /* The dimensions of this chunk of particles */
        const hsize_t start = (X_min * M * M);
        const hsize_t remaining = ptype->TotalNumber - start;
        const hsize_t chunk_size = MX * M * M;

        /* Sanity check */
        assert(MX * M * M <= remaining); //not out of bounds
        assert((local_X0 + local_NX) < N || MX * M * M == remaining); //exhaustive

        /* Allocate memory for our local chunk of particles */
        struct particle *parts = malloc(chunk_size * sizeof(struct particle));

        /* Generate the particles */
        int offset = 0;
        genParticlesFromGrid_local(&parts, &pars, &us, &cosmo, ptype, MX,
                                   X_min, offset, id_first_particle);

        /* We will also read slivers of the grids on both the left and the right */
        int extra_width = pars.NeighbourSliverSize;
        int left_sliver_X0 = wrap(local_X0 - extra_width, N);
        int right_sliver_X0 = wrap(local_X0 + local_NX, N);
        int left_sliver_NX = extra_width;
        int right_sliver_NX = extra_width;

        /* Sanity check: we don't want wrapping inside the slivers themselves  */
        assert(left_sliver_X0 + left_sliver_NX <= N);
        assert(right_sliver_X0 + right_sliver_NX <= N);
        assert(left_sliver_X0 >= 0);
        assert(right_sliver_X0 >= 0);

        printf("%03d: Local [%04d, %04d] left [%04d, %04d] right [%04d, %04d] particles [%04d, %04d]\n", rank, local_X0, local_X0 + local_NX, left_sliver_X0, left_sliver_X0 + left_sliver_NX, right_sliver_X0, right_sliver_X0 + right_sliver_NX, X_min, X_max);

        /* Package pointers and dimensions of the local slice and adjacent slivers */
        struct left_right_slice lrs;
        lrs.left_slice = fftw_alloc_real(left_sliver_NX * N * (N + 2));
        lrs.local_slice = fftw_alloc_real(local_NX * N * (N + 2));
        lrs.right_slice = fftw_alloc_real(right_sliver_NX * N * (N + 2));
        lrs.local_NX = local_NX;
        lrs.local_X0 = local_X0;
        lrs.left_NX = left_sliver_NX;
        lrs.left_X0 = left_sliver_X0;
        lrs.right_NX = right_sliver_NX;
        lrs.right_X0 = right_sliver_X0;

        /* Interpolating displacements at the pre-initial particle locations */
        /* For x, y, and z */
        const char letters[] = {'x', 'y', 'z'};
        for (int dir=0; dir<3; dir++) {
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c_%s%s", pars.OutputDirectory, GRID_NAME_DISPLACEMENT, letters[dir], ptype->Identifier, ".hdf5");

            /* Read our slice of the displacement grid */
            err = readField_MPI(lrs.local_slice, N, lrs.local_NX, lrs.local_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the left of the displacement grid */
            err = readField_MPI(lrs.left_slice, N, lrs.left_NX, lrs.left_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the right of the displacement grid */
            err = readField_MPI(lrs.right_slice, N, lrs.right_NX, lrs.right_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Displace the particles in this chunk */
            #pragma omp parallel for
            for (int i=0; i<chunk_size; i++) {
                /* Find the pre-initial (e.g. grid) locations */
                double x = parts[i].X;
                double y = parts[i].Y;
                double z = parts[i].Z;

                /* Find the displacement */
                double disp = gridTSC_dg(&lrs, x, y, z, boxlen, N);

                /* Displace the particles */
                if (dir == 0) {
                    parts[i].X -= disp;
                } else if (dir == 1) {
                    parts[i].Y -= disp;
                } else {
                    parts[i].Z -= disp;
                }
            }
        }

        /* Interpolating velocities at the displaced particle locations */
        /* For x, y, and z */
        for (int dir=0; dir<3; dir++) {
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%c_%s%s", pars.OutputDirectory, GRID_NAME_VELOCITY, letters[dir], ptype->Identifier, ".hdf5");
            // printf("Velocity field read from '%s'.\n", dbox_fname);

            /* Read our slice of the velocity grid */
            err = readField_MPI(lrs.local_slice, N, lrs.local_NX, lrs.local_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the left of the velocity grid */
            err = readField_MPI(lrs.left_slice, N, lrs.left_NX, lrs.left_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the right of the velocity grid */
            err = readField_MPI(lrs.right_slice, N, lrs.right_NX, lrs.right_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Assign velocities to the particles in this chunk */
            #pragma omp parallel for
            for (int i=0; i<chunk_size; i++) {
                /* Skip thermal particles if we only need the Firebolt sampler */
                if (ptype->UseFirebolt && (FIREBOLT_EXPLICIT_CHECKS == 0 ||
                    i % FIREBOLT_EXPLICIT_CHECKS != 0)) continue;

                /* Find the displaceed particle location */
                double x = parts[i].X;
                double y = parts[i].Y;
                double z = parts[i].Z;

                /* Find the velocity in the given direction */
                double vel = gridTSC_dg(&lrs, x, y, z, boxlen, N);

                /* Add the velocity component */
                if (dir == 0) {
                    parts[i].v_X = vel;
                } else if (dir == 1) {
                    parts[i].v_Y = vel;
                } else {
                    parts[i].v_Z = vel;
                }
            }
        }

        /* Add thermal motion */
        if (strcmp(ptype->ThermalMotionType, "") != 0) {
            /* Read the high-resolution configuration space density field */
            char dbox_fname[DEFAULT_STRING_LENGTH];
            sprintf(dbox_fname, "%s/%s_%s%s", pars.OutputDirectory, GRID_NAME_DENSITY, ptype->Identifier, ".hdf5");

            /* Read our slice of the density grid */
            err = readField_MPI(lrs.local_slice, N, lrs.local_NX, lrs.local_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the left of the density grid */
            err = readField_MPI(lrs.left_slice, N, lrs.left_NX, lrs.left_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Read a sliver on the right of the density grid */
            err = readField_MPI(lrs.right_slice, N, lrs.right_NX, lrs.right_X0, MPI_COMM_WORLD, dbox_fname);
            catch_error(err, "Error reading '%s'.\n", dbox_fname);

            /* Add thermal velocities to the particles in this chunk */
            for (int i=0; i<chunk_size; i++) {

                /* Resample as long necessary */
                char accept = 0;
                while (!accept) {
                    /* Draw a momentum in eV from the thermal distribution */
                    double p0_eV = samplerCustom(&thermal_sampler, &seed); //present-day momentum
                    double p_eV = p0_eV / a_ini; //redshifted momentum
                    thermal_draws++;

                    if (isnan(p_eV) || p_eV <= 0) {
                        printf("ERROR: invalid thermal momentum drawn: %e.\n", p_eV);
                        exit(1);
                    }

                    /* Convert to speed in internal units. Note that this is
                     * the spatial part of the relativistic 4-velocity. */
                    double V = p_eV / ptype->MicroscopicMass_eV * us.SpeedOfLight;

                    /* Generate a random point on the unit sphere using Gaussians */
                    double nx = sampleNorm(&seed);
                    double ny = sampleNorm(&seed);
                    double nz = sampleNorm(&seed);

                    /* And normalize */
                    double length = hypot(nx, hypot(ny, nz));
                    if (length > 0) {
                        nx /= length;
                        ny /= length;
                        nz /= length;
                    }

                    if (isnan(nx) || isnan(ny) || isnan(nz)) {
                        printf("ERROR: invalid random velocity v = [%e, %e, %e]\n", nx, ny, nz);
                        exit(1);
                    }

                    /* If desired, compute the linear theory perturbation */
                    if (ptype->UseFirebolt) {
                    #if(COMPILED_WITH_FIREBOLT)
                        /* Find the displaceed particle location */
                        double x = parts[i].X;
                        double y = parts[i].Y;
                        double z = parts[i].Z;

                        /* Compute the phase space density perturbation */
                        int mode = 2; //use all available orders of perturbations
                        double q = p0_eV / T_eV;
                        double Psi = evalDensity(firebolt.grids_ref, firebolt.q_size, firebolt.log_q_min, firebolt.log_q_max,
                                                 x, y, z, nx, ny, nz, q, mode);

                        /* The configuration space density perturbation as determined from the hi-res grid */
                        double density = gridTSC_dg(&lrs, x, y, z, boxlen, N);

                        if (isnan(Psi) || Psi <= -1) {
                            printf("ERROR: invalid perturbation to the probability.\n");
                            exit(1);
                        }

                        if (isnan(density) || density <= -1) {
                            printf("ERROR: invalid perturbation to the config space density.\n");
                            exit(1);
                        }

                        /* Rejection sampling, conditional on the configuration space density */
                        double base_accept = 1.0 - ptype->FireboltMaxPerturbation;
                        double p_accept = base_accept * (1 + Psi) / (1.0 + density);

                        if (p_accept > 1) {
                            printf("ERROR: rejection sampler encountered P(accept) > 1\t [q, Psi, P] = [%f, %e, %e].\n", q, Psi, p_accept);
                            exit(1);
                        }

                        /* Draw a uniform random number */
                        double u = sampleUniform(&seed);

                        /* Do we accept? */
                        if (u < p_accept) {
                            accept = 1;
                        }

                        /* If we accept, finish the calculation */
                        if (accept) {
                            /* For diagonstics, compare with hi-res grids occasionally */
                            if (FIREBOLT_EXPLICIT_CHECKS > 0 && i % FIREBOLT_EXPLICIT_CHECKS == 0) {
                                /* The gravitational component of the velocity (hasn't been updated yet)*/
                                double vx_g = parts[i].v_X;
                                double vy_g = parts[i].v_Y;
                                double vz_g = parts[i].v_Z;
                                double v_g = hypot(vx_g, hypot(vy_g, vz_g));

                                /* Compute the 0th order phase space density perturbation */
                                int mode_0 = 0; //use just the 0th order
                                double Psi_0 = evalDensity(firebolt.grids_ref, firebolt.q_size, firebolt.log_q_min, firebolt.log_q_max,
                                                           x, y, z, nx, ny, nz, q, mode_0);

                                /* Compute up to the 1st order phase space density perturbation in the gravitational flow direction */
                                int mode_1 = 1; //use just the 1st order
                                double Psi_1_g = evalDensity(firebolt.grids_ref, firebolt.q_size, firebolt.log_q_min, firebolt.log_q_max,
                                                             x, y, z, vx_g / v_g, vy_g / v_g, vz_g / v_g, q, mode_1);

                                /* Collect statistics to estimate corr(Psi, d) */
                                Psi_sum += Psi_0;
                                Psi2_sum += Psi_0 * Psi_0;
                                d_sum += density;
                                d2_sum += density * density;
                                Psi_d_sum += Psi_0 * density;

                                /* Determine whether the 1st order density perturbation makes sense */
                                if (Psi_1_g > 0)
                                correctly_oriented++;

                                /* Total number of checks */
                                explicit_Psi_checks++;
                            }

                            /* Set the velocity to be purely thermal. The gravitational flow
                             * speeds are already included in the acceptance probability. */
                             parts[i].v_X = nx * V;
                             parts[i].v_Y = ny * V;
                             parts[i].v_Z = nz * V;
                        }
                    #else
                    printf("ERROR: not compiled with Firebolt.\n");
                    exit(1);
                    #endif
                    } else {
                        /* Otherwise, always accept */
                        accept = 1;

                        /* And add the thermal velocities on top of the gravitational flow */
                        parts[i].v_X += nx * V;
                        parts[i].v_Y += ny * V;
                        parts[i].v_Z += nz * V;
                    }
                }
            }
        }

        /* Unit conversions */
        /* (...) */

        /* Make sure that particle coordinates wrap around */
        #pragma omp parallel for
        for (int i=0; i<chunk_size; i++) {
            parts[i].X = fwrap(parts[i].X, boxlen);
            parts[i].Y = fwrap(parts[i].Y, boxlen);
            parts[i].Z = fwrap(parts[i].Z, boxlen);
        }

        /* Before writing particle data to disk, we need to choose the
         * hyperslabs, i.e. the parts of memory where the data is stored.
         * In our case, these correspond to the chunks of particle
         * data, specified by a start and a dimensions vector.
         */

        /* Recall that multiple particle types can map into the same group.
         * For each particle type, we have already recorded the position of
         * its first particle in the group at ptype->PositionInExportGroup.
         */

        /* Create vector & scalar datapsace for smaller chunks of data */
        const hsize_t ch_vdims[2] = {chunk_size, 3};
        const hsize_t ch_sdims[2] = {chunk_size};
        hid_t h_ch_vspace = H5Screate_simple(vrank, ch_vdims, NULL);
        hid_t h_ch_sspace = H5Screate_simple(srank, ch_sdims, NULL);

        /* The start of this chunk, in the overall vector & scalar spaces */
        const hsize_t start_in_group = ptype->PositionInExportGroup + start;
        const hsize_t vstart[2] = {start_in_group, 0}; //always with the "x" coordinate
        const hsize_t sstart[1] = {start_in_group};

        /* Choose the corresponding hyperslabs inside the overall spaces */
        H5Sselect_hyperslab(h_vspace, H5S_SELECT_SET, vstart, NULL, ch_vdims, NULL);
        H5Sselect_hyperslab(h_sspace, H5S_SELECT_SET, sstart, NULL, ch_sdims, NULL);

        /* Unpack particle data into contiguous arrays */
        double *coords = malloc(3 * chunk_size * sizeof(double));
        double *vels = malloc(3 * chunk_size * sizeof(double));
        double *masses = malloc(1 * chunk_size * sizeof(double));
        long long *ids = malloc(1 * chunk_size * sizeof(long long));
        for (int i=0; i<chunk_size; i++) {
            coords[i * 3 + 0] = parts[i].X;
            coords[i * 3 + 1] = parts[i].Y;
            coords[i * 3 + 2] = parts[i].Z;
            vels[i * 3 + 0] = parts[i].v_X;
            vels[i * 3 + 1] = parts[i].v_Y;
            vels[i * 3 + 2] = parts[i].v_Z;
            masses[i] = parts[i].mass;
            ids[i] = parts[i].id;
        }

        /* Write coordinate data (vector) */
        h_data = H5Dopen(h_grp, "Coordinates", H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, coords);
        H5Dclose(h_data);
        free(coords);

        /* Write velocity data (vector) */
        h_data = H5Dopen(h_grp, "Velocities", H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_vspace, h_vspace, H5P_DEFAULT, vels);
        H5Dclose(h_data);
        free(vels);

        /* Write mass data (scalar) */
        h_data = H5Dopen(h_grp, "Masses", H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_DOUBLE, h_ch_sspace, h_sspace, H5P_DEFAULT, masses);
        H5Dclose(h_data);
        free(masses);

        /* Write particle id data (scalar) */
        h_data = H5Dopen(h_grp, "ParticleIDs", H5P_DEFAULT);
        H5Dwrite(h_data, H5T_NATIVE_LLONG, h_ch_sspace, h_sspace, H5P_DEFAULT, ids);
        H5Dclose(h_data);
        free(ids);

        /* Close the chunk-sized scalar and vector dataspaces */
        H5Sclose(h_ch_vspace);
        H5Sclose(h_ch_sspace);

        /* Free memory of the displacement and velocity grids */
        fftw_free(lrs.local_slice);
        fftw_free(lrs.left_slice);
        fftw_free(lrs.right_slice);

        /* Clean up some data structures if this particle type is thermal */
        if (strcmp(ptype->ThermalMotionType, "") != 0) {
            /* Clean the random sampler */
            cleanSampler(&thermal_sampler);

            /* Just for Firebolt diagnostics, compute some summary statistics */
            #if(COMPILED_WITH_FIREBOLT)
            if (ptype->UseFirebolt) {
                /* Sum the numer of thermal draws across all MPI ranks */
                if (rank == 0) {
                    MPI_Reduce(MPI_IN_PLACE, &thermal_draws, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Reduce(&thermal_draws, &thermal_draws, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                }

                /* Calculate the acceptance rate */
                if (rank == 0) {
                    double p_accept = (double) ptype->TotalNumber / thermal_draws;
                    message(rank, "\n");
                    message(rank, "Firebolt sampler acceptance rate: %.6f\n", p_accept);
                }

                /* Display other diagnostic summary statistics, computed for a
                 * small subsample of particles */
                if (FIREBOLT_EXPLICIT_CHECKS > 0) {
                    long long int num_stats[2] = {explicit_Psi_checks, correctly_oriented};
                    double Psi_d_stats[5] = {Psi_sum, Psi2_sum, d_sum, d2_sum, Psi_d_sum};

                    if (rank == 0) {
                        MPI_Reduce(MPI_IN_PLACE, num_stats, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(MPI_IN_PLACE, Psi_d_stats, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    } else {
                        MPI_Reduce(num_stats, num_stats, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
                        MPI_Reduce(Psi_d_stats, Psi_d_stats, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    }

                    /* Calculate the desired summary statistics */
                    if (rank == 0) {
                        double N_psi = (double) num_stats[0];
                        correctly_oriented = num_stats[1];

                        Psi_sum = Psi_d_stats[0];
                        Psi2_sum = Psi_d_stats[1];
                        d_sum = Psi_d_stats[2];
                        d2_sum = Psi_d_stats[3];
                        Psi_d_sum = Psi_d_stats[4];

                        double Psi_ss = N_psi * Psi2_sum - Psi_sum * Psi_sum;
                        double d_ss = N_psi * d2_sum - d_sum * d_sum;
                        double correlation = (N_psi * Psi_d_sum - Psi_sum * d_sum)
                                           / sqrt(Psi_ss * d_ss);
                        double p_oriented = correctly_oriented / N_psi;
                        message(rank, "\n");
                        message(rank, "Firebolt checks (number): N_checks = %e\n", N_psi);
                        message(rank, "Firebolt checks (density): corr(Psi_0, delta_hires) = %.6f\n", correlation);
                        message(rank, "Firebolt checks (velocity): P(Psi_1 > 0 along v_grav | theta_hires) = %.6f\n", p_oriented);
                    }
                }

                /* Clean the Firebolt Boltzmann code */
                cleanFirebolt();

                /* Free the small complex Gaussian random field */
                free(small_grf);
            }
            #endif
        }

        /* Close the scalar and vector dataspaces */
        H5Sclose(h_vspace);
        H5Sclose(h_sspace);

        /* Clean the particles up */
        cleanParticles(&parts, &pars, ptype);

        /* Close the group in the output file */
        H5Gclose(h_grp);
    }

    /* Close the output file */
    H5Fclose(h_out_file);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    /* Clean up */
    cleanExportGroups(&pars, &export_groups);
    cleanTypes(&pars, &types);
    cleanParams(&pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);

    /* Clean up the alternative perturbation file and associated spline */
    if (strcmp(pars.SecondPerturbFile, "") != 0) {
        cleanPerturb(&ptdat_alternative);
        cleanPerturbSpline(&spline_alternative);
    }

    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    message(rank, "\nTime elapsed: %.5f s\n", microsec/1e6);

}
