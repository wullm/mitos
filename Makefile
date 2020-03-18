#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES)
CFLAGS = -Wall

OBJECTS = lib/*.o

all:
	make minIni
	$(GCC) src/input.c -c -o lib/input.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/random.c -c -o lib/random.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/grf.c -c -o lib/grf.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/derivatives.c -c -o lib/derivatives.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/particle_types.c -c -o lib/particle_types.o $(INCLUDES) $(CFLAGS)
	$(GCC) src/dexm.c -o dexm $(INCLUDES) $(OBJECTS) $(LIBRARIES) $(CFLAGS)

minIni:
	cd parser && make

check:
	cd tests && make
