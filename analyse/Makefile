#Compiler options
GCC = gcc

#Libraries
INI_PARSER = ../parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

FIREBOLT_PATH = /home/qvgd89/firebolt2/firebolt
FIREBOLT_INCLUDES += -I$(FIREBOLT_PATH)/include
FIREBOLT_LIBRARIES += -L$(FIREBOLT_PATH) -lfirebolt -Wl,-rpath=$(FIREBOLT_PATH)

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES) $(FIREBOLT_LIBRARIES)
CFLAGS = -Wall -fopenmp -march=native -O4

OBJECTS = ../lib/*.o

all:
	$(GCC) src/dexm_read.c -o dexm_read $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
	$(GCC) src/dexm_render.c -o dexm_render $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)
