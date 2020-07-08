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
#include "../include/header.h"

int writeHeaderAttributes(struct params *pars, struct cosmology *cosmo,
                          struct units *us, struct particle_type **types,
                          hid_t h_file) {

    /* Create the Header group */
    hid_t h_grp = H5Gcreate(h_file, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for BoxSize attribute */
    const hsize_t arank = 1;
    const hsize_t adims_three[1] = {3}; //3D space
    hid_t h_aspace = H5Screate_simple(arank, adims_three, NULL);

    /* Create the BoxSize attribute and write the data */
    hid_t h_attr = H5Acreate1(h_grp, "BoxSize", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    double boxlen = pars->BoxLen;
    double boxsize[3] = {boxlen, boxlen, boxlen};
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, boxsize);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to scalar value attributes */
    const hsize_t adims_single[1] = {1};
    H5Sset_extent_simple(h_aspace, arank, adims_single, NULL);

    /* Create the Dimension attribute and write the data */
    int dimension = 3;
    h_attr = H5Acreate1(h_grp, "Dimension", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &dimension);
    H5Aclose(h_attr);

    /* Create the Redshift attribute and write the data */
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &cosmo->z_ini);
    H5Aclose(h_attr);

    /* Create the Flag_Entropy_ICs attribute and write the data */
    int flag_entropy = 0;
    h_attr = H5Acreate1(h_grp, "Flag_Entropy_ICs", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &flag_entropy);
    H5Aclose(h_attr);

    /* Create the NumFilesPerSnapshot attribute and write the data */
    int num_files_per_snapshot = 1;
    h_attr = H5Acreate1(h_grp, "NumFilesPerSnapshot", H5T_NATIVE_INT, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_INT, &num_files_per_snapshot);
    H5Aclose(h_attr);

    /* Change dataspace dimensions to particle type attributes */
    const hsize_t adims_pt[1] = {7}; //particle type 0-6
    H5Sset_extent_simple(h_aspace, arank, adims_pt, NULL);

    /* Collect particle type attributes using the ExportNames */
    long long int numparts[7] = {0, 0, 0, 0, 0, 0, 0};
    long long int numparts_high_word[7] = {0, 0, 0, 0, 0, 0, 0}; //not used, so use zeros
    double mass_table[7] = {0., 0., 0., 0., 0., 0., 0.}; //not used, so use zeros
    for (int i=0; i<7; i++) {
        char ptype_name[40];
        sprintf(ptype_name, "PartType%d", i);

        /* Find a particle type with this export name */
        for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
            struct particle_type *ptype = *types + pti;
            if (strcmp(ptype->ExportName, ptype_name) == 0) {
                numparts[i] += ptype->TotalNumber;
            }
        }
    }

    /* Create the NumPart_Total attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts);
    H5Aclose(h_attr);

    /* Create the NumPart_Total_HighWord attribute and write the data */
    h_attr = H5Acreate1(h_grp, "NumPart_Total_HighWord", H5T_NATIVE_LONG, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_LONG, numparts_high_word);
    H5Aclose(h_attr);

    /* Create the MassTable attribute and write the data */
    h_attr = H5Acreate1(h_grp, "MassTable", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, mass_table);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Header group */
    H5Gclose(h_grp);

    /* Create the Cosmology group */
    h_grp = H5Gcreate(h_file, "/Cosmology", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Create the Redshift attribute and write the data */
    h_attr = H5Acreate1(h_grp, "Redshift", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &cosmo->z_ini);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);

    /* Create the Units group */
    h_grp = H5Gcreate(h_file, "/Units", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    /* Create dataspace for scalar value attributes */
    h_aspace = H5Screate_simple(arank, adims_single, NULL);

    /* Determine the units used */
    double unit_mass_cgs = us->UnitMassKilogram * 1000;
    double unit_length_cgs = us->UnitLengthMetres * 100;
    double unit_time_cgs = us->UnitTimeSeconds;
    double unit_temperature_cgs = us->UnitTemperatureKelvin;
    double unit_current_cgs = us->UnitCurrentAmpere;

    /* Write the internal unit system */
    h_attr = H5Acreate1(h_grp, "Unit mass in cgs (U_M)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_mass_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit length in cgs (U_L)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_length_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit time in cgs (U_t)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_time_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit temperature in cgs (U_T)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_temperature_cgs);
    H5Aclose(h_attr);

    h_attr = H5Acreate1(h_grp, "Unit current in cgs (U_I)", H5T_NATIVE_DOUBLE, h_aspace, H5P_DEFAULT);
    H5Awrite(h_attr, H5T_NATIVE_DOUBLE, &unit_current_cgs);
    H5Aclose(h_attr);

    /* Close the attribute dataspace */
    H5Sclose(h_aspace);

    /* Close the Cosmology group */
    H5Gclose(h_grp);


    return 0;
}


int writeSwiftParameterFile(struct params *pars, struct cosmology *cosmo,
                            struct units *us, struct particle_type **types,
                            struct perturb_params *ptpars, const char *fname) {
    /* Open a text file */
    FILE *f = fopen(fname, "w+");

    /* Determine the units used */
    double unit_mass_cgs = us->UnitMassKilogram * 1000;
    double unit_length_cgs = us->UnitLengthMetres * 100;
    double unit_time_cgs = us->UnitTimeSeconds;
    double unit_velocity_cgs = unit_length_cgs/unit_time_cgs;
    double unit_temperature_cgs = us->UnitTemperatureKelvin;
    double unit_current_cgs = us->UnitCurrentAmpere;

    fprintf(f, "InternalUnitSystem:\n");
    fprintf(f, "  UnitMass_in_cgs:\t%.10e\n", unit_mass_cgs);
    fprintf(f, "  UnitLength_in_cgs:\t%.10e\n", unit_length_cgs);
    fprintf(f, "  UnitVelocity_in_cgs:\t%.10e\n", unit_velocity_cgs);
    fprintf(f, "  UnitCurrent_in_cgs:\t%.10e\n", unit_current_cgs);
    fprintf(f, "  UnitTemp_in_cgs:\t%.10e\n", unit_temperature_cgs);
    fprintf(f, "\n");

    /* Determine cosmological parameters */
    double a_first = 1.0 / (cosmo->z_ini + 1.0);
    double T_CMB = ptpars->T_CMB;
    /* SWIFT only supports one neutrino temperature, so use the first one */
    double T_nu = ptpars->T_ncdm[0] * T_CMB;

    fprintf(f, "Cosmology:\n");
    fprintf(f, "  Omega_m:\t%.10f\n", ptpars->Omega_m);
    fprintf(f, "  Omega_b:\t%.10f\n", ptpars->Omega_b);
    fprintf(f, "  Omega_lambda:\t%.10f\n", ptpars->Omega_lambda);
    fprintf(f, "  Omega_k:\t%.10f\n", ptpars->Omega_k);
    fprintf(f, "  Omega_ur:\t%.10f\n", ptpars->Omega_ur);
    fprintf(f, "  h:\t\t%.10f\n", ptpars->h);
    fprintf(f, "  a_begin:\t%.10f # z = %f\n", a_first, cosmo->z_ini);
    fprintf(f, "  a_end:\t%.10f\n", 1.0);
    fprintf(f, "  T_CMB:\t%.10f\n", T_CMB);
    fprintf(f, "  T_nu:\t\t%.10f\n", T_nu);

    /* Print the number of neutrino species and their masses */
    fprintf(f, "  N_nu:\t\t%d\n", ptpars->N_ncdm);
    fprintf(f, "  M_nu:\t\t");
    for (int i=0; i<ptpars->N_ncdm; i++) {
        fprintf(f, "%.10f", ptpars->M_ncdm_eV[i]);
        if (i < ptpars->N_ncdm - 1) {
            fprintf(f, ", ");
        } else {
            fprintf(f, "\n");
        }
    }
    fprintf(f, "\n");

    /* Some reasonable SWIFT parameters that can be easily changed */
    double dt_min = 1e-10;
    double dt_max = 1e-2;

    fprintf(f, "TimeIntegration:\n");
    fprintf(f, "  dt_min:\t%.4e\n", dt_min);
    fprintf(f, "  dt_max:\t%.4e\n", dt_max);
    fprintf(f, "\n");

    /* Some reasonable SWIFT parameters that can be easily changed */
    int max_top_level_cells = 16;
    int cell_split_size = 100;

    fprintf(f, "Scheduler:\n");
    fprintf(f, "  max_top_level_cells:\t%d\n", max_top_level_cells);
    fprintf(f, "  cell_split_size:\t%d\n", cell_split_size);
    fprintf(f, "\n");

    /* Some reasonable SWIFT parameters that can be easily changed */
    double delta_time = 1.25;
    int compression = 4;
    char snapshot_basename[10] = "box";
    double delta_hours = 1.0;

    fprintf(f, "Snapshots:\n");
    fprintf(f, "  scale_factor_first:\t%.10f # z = %f\n", a_first, cosmo->z_ini);
    fprintf(f, "  delta_time:\t\t%.4f\n", delta_time);
    fprintf(f, "  basename:\t\t%s\n", snapshot_basename);
    fprintf(f, "  compression:\t\t%d\n", compression);
    fprintf(f, "\n");
    fprintf(f, "Statistics:\n");
    fprintf(f, "  scale_factor_first:\t%.10f # z = %f\n", a_first, cosmo->z_ini);
    fprintf(f, "  delta_time:\t\t%.4f\n", delta_time);
    fprintf(f, "\n");
    fprintf(f, "Restarts:\n");
    fprintf(f, "  delta_hours:\t%.2f\n", delta_hours);
    fprintf(f, "\n");

    /* Some reasonable SWIFT parameters that can be easily changed */
    int periodic = 1;
    int dithering = 0;
    double eta = 0.025;
    double theta = 0.5;
    char MAC[10] = "geometric";

    fprintf(f, "InitialConditions:\n");
    fprintf(f, "  file_name:\t%s\n", pars->OutputFilename);
    fprintf(f, "  periodic:\t%d\n", periodic);
    fprintf(f, "\n");
    fprintf(f, "Gravity:\n");
    fprintf(f, "  mesh_side_length:\t%d\n", pars->GridSize);
    fprintf(f, "  dithering:\t\t%d\n", dithering);
    fprintf(f, "  MAC:\t\t\t%s\n", MAC);
    fprintf(f, "  eta:\t\t\t%.5f\n", eta);
    fprintf(f, "  theta_cr:\t\t%.5f\n", theta);

    /* Collect the number of particles per ExportNames */
    long long int numparts[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i=0; i<7; i++) {
        char ptype_name[40];
        sprintf(ptype_name, "PartType%d", i);

        /* Find a particle type with this export name */
        for (int pti = 0; pti < pars->NumParticleTypes; pti++) {
            struct particle_type *ptype = *types + pti;
            if (strcmp(ptype->ExportName, ptype_name) == 0) {
                numparts[i] += ptype->TotalNumber;
            }
        }
    }

    /* Compute the mean particle separation for each type */
    double particle_dist[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i=0; i<7; i++) {
        if (numparts[i] > 0) {
            particle_dist[i] = pars->BoxLen / cbrt(numparts[i]);
        }
    }

    double softening_factor = 1.0 / 25.0;
    char ofstring[50] = "of the mean inter-particle separation";

    /* Find the mean particle sepatarion of PartType1 */
    double softening_DM = particle_dist[1] * softening_factor;
    double softening_nu = particle_dist[6] * softening_factor;
    double softening_b = particle_dist[0] * softening_factor;

    if (softening_DM > 0) {
        fprintf(f, "  comoving_DM_softening:\t\t%.10f \t# %.4f %s\n", softening_DM, softening_factor, ofstring);
        fprintf(f, "  max_physical_DM_softening:\t\t%.10f \t# %.4f %s\n", softening_DM, softening_factor, ofstring);
    }
    if (softening_nu > 0) {
        fprintf(f, "  comoving_nu_softening:\t\t%.10f \t# %.4f %s\n", softening_nu, softening_factor, ofstring);
        fprintf(f, "  max_physical_nu_softening:\t\t%.10f \t# %.4f %s\n", softening_nu, softening_factor, ofstring);
    }
    if (softening_b > 0) {
        fprintf(f, "  comoving_baryon_softening:\t\t%.10f \t# %.4f %s\n", softening_b, softening_factor, ofstring);
        fprintf(f, "  max_physical_baryon_softening:\t%.10f \t# %.4f %s\n", softening_b, softening_factor, ofstring);
    }


    /* Close the file */
    fclose(f);

    return 0;
}
