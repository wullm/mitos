#Compiler options
GCC = mpicc

#Libraries
INI_PARSER = ../parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3 -lfftw3_omp -lfftw3_mpi
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -I/usr/include/hdf5/openmpi

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall -Wshadow=global -fopenmp -march=native -O4
LDFLAGS =

OBJECTS = ../lib/input.o ../lib/output.o ../lib/input_mpi.o ../lib/output_mpi.o ../lib/fft.o ../lib/calc_powerspec.o ../lib/grids_interp.o ../lib/particle_types.o ../lib/titles.o ../lib/distributed_grid.o ../lib/random.o ../lib/perturb_data.o ../lib/perturb_spline.o ../lib/primordial.o

all:
	$(GCC) src/mitos_read.c -o mitos_read $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)
	$(GCC) src/mitos_box.c -o mitos_box $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)
	$(GCC) src/mitos_cross_spec.c -o mitos_cross_spec $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)
	$(GCC) src/mitos_profiles.c -o mitos_profiles $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)
	$(GCC) src/mitos_mesh_profiles.c -o mitos_mesh_profiles $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)
	$(GCC) src/mitos_render.c -o mitos_render $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
	$(GCC) src/mitos_gauss_purifier.c -o mitos_gauss_purifier $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
	$(GCC) src/mitos_vel3.c -o mitos_vel3 $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
