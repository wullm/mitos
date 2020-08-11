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

#ifndef OUTPUT_MPI_H
#define OUTPUT_MPI_H

#include <mpi.h>
#include <hdf5.h>

hid_t openFile_MPI(MPI_Comm comm, const char *fname);
hid_t createFile_MPI(MPI_Comm comm, const char *fname);
int createFieldGroup_MPI(int N, int NX, hid_t h_file);

int writeFieldFile_MPI(double *data, int N, int NX, int X0, double boxlen, MPI_Comm comm, const char *fname);
int prepareFieldFile_MPI(int N, int NX, double boxlen, MPI_Comm comm, const char *fname);
int writeData_MPI(const double *data, int N, int NX, int X0,
                  MPI_Comm comm, const hid_t h_file);

#endif
