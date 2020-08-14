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
CFLAGS = -Wall -fopenmp -march=native -O4

OBJECTS = ../lib/*.o

all:
	$(GCC) src/mitos_read.c -o mitos_read $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
	#$(GCC) src/mitos_render.c -o mitos_render $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
