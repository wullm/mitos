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

#include "../include/mitos.h"

#define printheader(s) printf("\n%s%s%s\n", TXT_BLUE, s, TXT_RESET);

const char *fname;

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int MPI_Rank_ID, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_Rank_ID);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);

    /* Read options */
    const char *fname = argv[1];
    if (MPI_Rank_ID == 0) {
        printheader("Mitos Initial Condition Generator");
        printf("The parameter file is '%s'\n", fname);
    }

    /* Timer */
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    /* Mitos structuress */
    struct params pars;
    struct units us;
    struct particle_type *types = NULL;
    struct export_group *export_groups = NULL;
    struct cosmology cosmo;
    struct perturb_data ptdat;
    struct perturb_spline spline;
    struct perturb_params ptpars;

    /* Read parameter file for parameters, units, and cosmological values */
    readParams(&pars, fname);
    readUnits(&us, fname);
    readCosmology(&cosmo, &us, fname);

    if (MPI_Rank_ID == 0) {
        printf("The output directory is '%s'.\n", pars.OutputDirectory);
        printf("Creating initial conditions for '%s'.\n", pars.Name);
    }

    /* Allow multi-threaded FFT */
    // fftw_init_threads();
    // fftw_plan_with_nthreads(1);
    // printf("Running with %d threads.\n", pars.Threads);

    /* Read out particle types from the parameter file */
    readTypes(&pars, &types, fname);

    /* Match particle types with export groups */
    fillExportGroups(&pars, &types, &export_groups);

    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat);
    readPerturbParams(&pars, &us, &ptpars);

    /* Do a sanity check */
    if (fabs(cosmo.h - ptpars.h) / cosmo.h > 1e-5) {
        printf("ERROR: h from parameter file does not match perturbation file.\n");
        exit (1);
    }

    /* Merge cdm & baryons into one set of transfer functions (replacing cdm) */
    if (pars.MergeDarkMatterBaryons) {
        if (MPI_Rank_ID == 0)
        printheader("Merging cdm & baryon transfer functions, replacing cdm.");

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

        if (MPI_Rank_ID == 0)
        printf("Using weights [w_cdm, w_b] = [%f, %f]\n", weight_cdm, weight_b);

        /* Merge the density & velocity transfer runctions, replacing cdm */
        mergeTransferFunctions(&ptdat, "d_cdm", "d_b", weight_cdm, weight_b);
        mergeTransferFunctions(&ptdat, "t_cdm", "t_b", weight_cdm, weight_b);
        /* Merge the background densities, replacing cdm */
        mergeBackgroundDensities(&ptdat, "d_cdm", "d_b", 1.0, 1.0); //replace with sum
    }

    /* Initialize the interpolation spline for the perturbation data */
    initPerturbSpline(&spline, DEFAULT_K_ACC_TABLE_SIZE, &ptdat);

    /* Seed the random number generator */
    rng_state seed = rand_uint64_init(pars.Seed + MPI_Rank_ID);

    /* Determine the starting conformal time */
    cosmo.log_tau_ini = perturbLogTauAtRedshift(&spline, cosmo.z_ini);

    if (MPI_Rank_ID == 0) {
        printheader("Settings");
        printf("Random numbers\t\t [seed] = [%ld]\n", pars.Seed);
        printf("Starting time\t\t [z, tau] = [%.2f, %.2f U_T]\n", cosmo.z_ini, exp(cosmo.log_tau_ini));
        printf("Primordial power\t [A_s, n_s, k_pivot] = [%.4e, %.4f, %.4f U_L]\n", cosmo.A_s, cosmo.n_s, cosmo.k_pivot);
        printf("\n");

        printheader("Requested Particle Types");
        for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
            /* The current particle type */
            struct particle_type *ptype = types + pti;
            printf("Particle type '%s' (N^3 = %d^3).\n", ptype->Identifier, ptype->CubeRootNumber);
        }
    }

    /* Create Gaussian random field */
    const int N = pars.GridSize;
    const double boxlen = pars.BoxLen;

    long int local_NX;     //the portion on this rank will be NX * N * (N/2 + 1)
    long int local_X0;     //location along the X-axis of the portion
    long int local_size;   //number of elements on this rank

    /* Determine the portion allocated on this rank, sliced along the X-axis */
    local_size = fftw_mpi_local_size_3d(N, N, N/2+1, MPI_COMM_WORLD, &local_NX, &local_X0);

    /* Allocate a portion of the Gaussian random field */
    fftw_complex *grf = fftw_alloc_complex(local_size);
    double *box = fftw_alloc_real(2*local_size);

    /* Create MPI FFTW plan */
    fftw_plan r2c_mpi = fftw_mpi_plan_dft_c2r_3d(N, N, N, grf, box, MPI_COMM_WORLD, FFTW_ESTIMATE);

    // printf("Local size is %ld, local n0 is %ld %ld\n", local_size, local_NX, local_X0);

    /* Allocate 3D array */
    // fftw_complex *grf = (fftw_complex*) fftw_malloc(N*N*(N/2+1)*sizeof(fftw_complex));

    /* Generate a complex Hermitian Gaussian random field */
    if (MPI_Rank_ID == 0) printheader("Generating Primordial Fluctuations");
    generate_complex_grf(grf, N, local_NX, local_X0, boxlen, &seed);
    // enforce_hermiticity(grf, N, local_NX, local_X0, boxlen, &seed, MPI_COMM_WORLD);

    /* Apply the bare power spectrum, without any transfer functions */
    fft_apply_kernel(grf, grf, N, local_NX, local_X0, boxlen, kernel_power_no_transfer, &cosmo);

    /* Execute the Fourier transform and normalize */
    fft_execute(r2c_mpi);
    fft_normalize_c2r(box, N, local_NX, local_X0, boxlen);

    /* Free up memory */
    fftw_free(grf);
    fftw_destroy_plan(r2c_mpi);

    /* Export the real GRF */
    char grf_fname[DEFAULT_STRING_LENGTH];
    sprintf(grf_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");
    int err = writeFieldFile_MPI(box, N, local_NX, local_X0, boxlen, MPI_COMM_WORLD, grf_fname);
    if (err > 0) printf("Error while writing '%s'.\n", fname);
    if (MPI_Rank_ID == 0) printf("Pure Gaussian Random Field exported to '%s'.\n", grf_fname);

    /* Free memory of the real-space GRF */
    fftw_free(box);

    // /* Convert from complex phases to real Gaussian variates and export the box */
    // char grf_fname[DEFAULT_STRING_LENGTH];
    // sprintf(grf_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN, ".hdf5");
    // fft_c2r_export_and_free(grf, N, boxlen, grf_fname); //frees grf
    // printf("Pure Gaussian Random Field exported to '%s'.\n", grf_fname);

    // /* Create a smaller (zoomed out) copy of the Gaussian random field */
    // if (pars.SmallGridSize > 0) {
    //     char small_fname[DEFAULT_STRING_LENGTH];
    //     sprintf(small_fname, "%s/%s%s", pars.OutputDirectory, GRID_NAME_GAUSSIAN_SMALL, ".hdf5");
    //     int errs = shrinkGridExport(pars.SmallGridSize, small_fname, grf_fname);
    //     if (errs > 0) exit(1);
    //     printf("Smaller copy of the Gaussian Random Field exported to '%s'.\n", small_fname);
    // }


    /* Retrieve background densities from the perturbations data file */
    if (MPI_Rank_ID == 0) printheader("Fetching Background Densities");
    retrieveDensities(&pars, &cosmo, &types, &ptdat);
    retrieveMicroMasses(&pars, &cosmo, &types, &ptpars);

    /* For each particle type, fetch the user-defined density function title */
    if (MPI_Rank_ID == 0) printheader("Fetching Density Perturbations");
    char **function_titles = malloc(pars.NumParticleTypes * sizeof(char*));
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;
        function_titles[pti] = ptype->TransferFunctionDensity;
        if (MPI_Rank_ID == 0) printf("Particle type '%s' uses density vector '%s'.\n", Identifier, function_titles[pti]);
    }

    /* Generate the density grids */
    if (MPI_Rank_ID == 0) printheader("Generating Density Grids");
    err = generatePerturbationGrids(&pars, &us, &cosmo, &spline, types, function_titles, grf_fname, GRID_NAME_DENSITY, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    /* For each particle type, fetch the user-defined energy flux function title */
    printheader("Fetching Energy Flux Perturbations");
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        struct particle_type *ptype = types + pti;
        const char *Identifier = ptype->Identifier;
        function_titles[pti] = ptype->TransferFunctionVelocity;
        printf("Particle type '%s' uses energy flux vector '%s'.\n", Identifier, function_titles[pti]);
    }

    /* Generate the energy flux (velocity divergence theta) grids */
    printheader("Generating Energy Flux Fields");
    err = generatePerturbationGrids(&pars, &us, &cosmo, &spline, types, function_titles, grf_fname, GRID_NAME_THETA, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    /* Get rid of the perturbation vector function titles */
    free(function_titles);

    /* Compute SPT grids */
    printheader("Computing SPT Corrections");
    err = computePerturbedGrids(&pars, &us, &cosmo, types, GRID_NAME_DENSITY, GRID_NAME_THETA);
    if (err > 0) exit(1);

    /* Compute the potential grids */
    printheader("Computing Gravitational Potentials");
    err = computePotentialGrids(&pars, &us, &cosmo, types, GRID_NAME_DENSITY, GRID_NAME_POTENTIAL, /* withELPT = */ 1, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    /* Compute derivatives of the potential grids */
    printheader("Computing Potential Derivatives (Displacements)");
    err = computeGridDerivatives(&pars, &us, &cosmo, types, GRID_NAME_POTENTIAL, GRID_NAME_DISPLACEMENT, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);

    /* Compute the energy flux potential grids */
    printheader("Computing Energy Flux Potentials");
    err = computePotentialGrids(&pars, &us, &cosmo, types, GRID_NAME_THETA, GRID_NAME_THETA_POTENTIAL, /* withELPT = */ 0, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    /* Compute derivatives of the energy flux grids */
    printheader("Computing Energy Flux Derivatives (Velocities)");
    err = computeGridDerivatives(&pars, &us, &cosmo, types, GRID_NAME_THETA_POTENTIAL, GRID_NAME_VELOCITY, N, local_NX, local_X0, local_size, boxlen, MPI_COMM_WORLD);
    if (err > 0) exit(1);

    /* Create the beginning of a SWIFT parameter file */
    printheader("Creating SWIFT Parameter File");
    char out_par_fname[DEFAULT_STRING_LENGTH];
    sprintf(out_par_fname, "%s/%s", pars.OutputDirectory, pars.SwiftParamFilename);
    printf("Creating output file '%s'.\n", out_par_fname);
    writeSwiftParameterFile(&pars, &cosmo, &us, &types, &ptpars, out_par_fname);

    /* Name of the main output file containing the initial conditions */
    printheader("Initializing Output File");
    char out_fname[DEFAULT_STRING_LENGTH];
    sprintf(out_fname, "%s/%s", pars.OutputDirectory, pars.OutputFilename);
    printf("Creating output file '%s'.\n", out_fname);

    /* Create the output file */
    hid_t h_out_file = H5Fcreate(out_fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

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


    /* For each user-defined particle type */
    for (int pti = 0; pti < pars.NumParticleTypes; pti++) {
        /* The current particle type */
        struct particle_type *ptype = types + pti;

        char header[50];
        sprintf(header, "Generating Particle Type '%s'.", ptype->Identifier);
        printheader(header);

        /* Skip empty particle types */
        if (ptype->TotalNumber <= 0) {
            printf("No particles requested.\n");
            continue;
        }

        /* ID of the first particle of this type */
        const long long int id_first_particle = ptype->FirstID;

        /* Random sampler used for thermal species */
        struct sampler thermal_sampler;

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
            double M_eV = ptype->MicroscopicMass_eV;
            /* Convert the temperature to electronVolts */
            double T_eV = ptype->MicroscopyTemperature * us.kBoltzmann / us.ElectronVolt;
            /* Chemical potential */
            double mu_eV = 0;

            /* Rescale the domain */
            xl *= T_eV;
            xr *= T_eV;

            /* Initialize the sampler */
            double thermal_params[2] = {T_eV, mu_eV};

            int err = initSampler(&thermal_sampler, function, xl, xr, thermal_params);
            if (err > 0) {
                printf("Error initializing the thermal motion sampler.\n");
                exit(1);
            }

            printf("Thermal motion: %s with [M, T] = [%e eV, %e eV].\n",
                    ptype->ThermalMotionType, M_eV, T_eV);
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

        /* Allocate enough memory for one chunk of particles */
        struct particle *parts;
        allocParticles(&parts, &pars, ptype);

        /* Allocate memory for the displacement and velocity grids */
        double *grid = malloc(N*N*N * sizeof(double));

        /* For each chunk, generate and store the particles */
        for (int chunk=0; chunk<ptype->Chunks; chunk++) {
            /* The dimensions of this chunk */
            const hsize_t start = chunk * ptype->ChunkSize;
            const hsize_t remaining = ptype->TotalNumber - start;
            const hsize_t chunk_size = (hsize_t) fmin(ptype->ChunkSize, remaining);

            printf("Generating chunk %d.\n", chunk);
            genParticles_FromGrid(&parts, &pars, &us, &cosmo, ptype, chunk, id_first_particle);

            /* Interpolating displacements at the pre-initial particle locations */
            /* For x, y, and z */
            const char letters[] = {'x', 'y', 'z'};
            for (int dir=0; dir<3; dir++) {
                char dbox_fname[DEFAULT_STRING_LENGTH];
                sprintf(dbox_fname, "%s/%s_%c_%s%s", pars.OutputDirectory, GRID_NAME_DISPLACEMENT, letters[dir], ptype->Identifier, ".hdf5");
                // printf("Displacement field read from '%s'.\n", dbox_fname);
                int err = readGRF_inPlace_H5(grid, dbox_fname);
                if (err > 0) exit(1);

                /* Displace the particles in this chunk */
                #pragma omp parallel for
                for (int i=0; i<chunk_size; i++) {
                    /* Find the pre-initial (e.g. grid) locations */
                    double x = parts[i].X;
                    double y = parts[i].Y;
                    double z = parts[i].Z;

                    /* Find the displacement */
                    double disp = gridPCS(grid, N, boxlen, x, y, z);

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
                int err = readGRF_inPlace_H5(grid, dbox_fname);
                if (err > 0) exit(1);

                /* Assign velocities to the particles in this chunk */
                #pragma omp parallel for
                for (int i=0; i<chunk_size; i++) {
                    /* Find the displaceed particle location */
                    double x = parts[i].X;
                    double y = parts[i].Y;
                    double z = parts[i].Z;

                    /* Find the velocity in the given direction */
                    double vel = gridPCS(grid, N, boxlen, x, y, z);

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
                const double a_ini = 1.0 / (cosmo.z_ini + 1.0);

                /* Add thermal velocities to the particles in this chunk */
                for (int i=0; i<chunk_size; i++) {
                    /* Draw a momentum in eV from the thermal distribution */
                    double p0_eV = samplerCustom(&thermal_sampler, &seed); //present-day momentum
                    double p_eV = p0_eV / a_ini; //redshifted momentum

                    if (isnan(p_eV) || p_eV <= 0) {
                        printf("ERROR: invalid thermal momentum drawn: %e.\n", p_eV);
                        exit(1);
                    }

                    /* Convert to speed in internal units. Note that this is
                     * the spatial part of the relativistic 4-velocity. */
                    double V = p_eV / ptype->MicroscopicMass_eV * us.SpeedOfLight;

                    /* Generate a random point on the unit sphere using Gaussians */
                    double x = sampleNorm(&seed);
                    double y = sampleNorm(&seed);
                    double z = sampleNorm(&seed);

                    /* And normalize */
                    double length = hypot(x, hypot(y, z));
                    if (length > 0) {
                        x /= length;
                        y /= length;
                        z /= length;
                    }

                    if (isnan(x) || isnan(y) || isnan(z)) {
                        printf("ERROR: invalid random velocity v = [%e, %e, %e]\n", x, y, z);
                        exit(1);
                    }

                    /* Add the thermal velocities */
                    parts[i].v_X += x * V;
                    parts[i].v_Y += y * V;
                    parts[i].v_Z += z * V;
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

        }

        /* Free memory of the displacement and velocity grids */
        free(grid);

        /* Clean up the random sampler if this particle type is thermal */
        if (strcmp(ptype->ThermalMotionType, "") != 0) {
            cleanSampler(&thermal_sampler);
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

    /* Clean up FFTW structures */
    // fftw_cleanup_threads();

    /* Done with MPI parallelization */
    MPI_Finalize();

    /* Clean up */
    cleanExportGroups(&pars, &export_groups);
    cleanTypes(&pars, &types);
    cleanParams(&pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);

    /* Release the interpolation splines */
    cleanPerturbSpline(&spline);

    /* Timer */
    gettimeofday(&stop, NULL);
    long unsigned microsec = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
    printf("\nTime elapsed: %.5f s\n", microsec/1e6);

}
