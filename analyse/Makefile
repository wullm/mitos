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
	
PROGRAMS = mitos_read mitos_half_read mitos_box mitos_cross_spec mitos_profiles mitos_mesh_profiles mitos_render mitos_gauss_purifier mitos_vel3 mitos_veloc_bias mitos_halo_vel3
	
all: $(PROGRAMS)

$(PROGRAMS): % : src/%.c $(OBJECTS)
	$(GCC) src/$@.c -o $@ $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(LDFLAGS)