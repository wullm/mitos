#Compiler options
GCC = gcc

#Libraries
INI_PARSER = ../parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES)
CFLAGS = -Wall

OBJECTS = ../lib/*.o

all:
	@#$(GCC) test_minIni.c -o test_minIni $(INI_PARSER)
	@#@./test_minIni

	$(GCC) test_input.c -o test_input $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_input

	$(GCC) test_random.c -o test_random $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_random

	$(GCC) test_fft.c -o test_fft $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test.box
	@./test_fft

	$(GCC) test_poisson.c -o test_poisson $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_poisson

	$(GCC) test_hdf5.c -o test_hdf5 $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f tests/test.hdf5
	@./test_hdf5
