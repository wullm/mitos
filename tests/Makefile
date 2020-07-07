#Compiler options
GCC = gcc

#Libraries
INI_PARSER = ../parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
HDF5_LIBRARIES = -lhdf5
GSL_LIBRARIES = -lgsl -lgslcblas

GSL_INCLUDES =

HDF5_INCLUDES += -I/usr/lib/x86_64-linux-gnu/hdf5/serial/include
HDF5_LIBRARIES += -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial

#Putting it together
INCLUDES = $(HDF5_INCLUDES) $(GSL_INCLUDES)
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES) $(HDF5_LIBRARIES) $(GSL_LIBRARIES)
CFLAGS = -Wall

OBJECTS = ../lib/*.o

all:
	@#$(GCC) test_minIni.c -o test_minIni $(INI_PARSER)
	@#@./test_minIni


	$(GCC) test_transfer_render.c -o test_transfer_render $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test_trr.box
	@./test_transfer_render

	$(GCC) test_input.c -o test_input $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_input

	$(GCC) test_random.c -o test_random $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_random

	$(GCC) test_fft.c -o test_fft $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test.box
	@./test_fft

	$(GCC) test_poisson.c -o test_poisson $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_poisson

	$(GCC) derivatives.c -c -o derivatives.o $(INCLUDES) $(CFLAGS)
	$(GCC) test_calc_powerspec.c -o test_calc_powerspec derivatives.o $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test_ps.box
	@./test_calc_powerspec

	$(GCC) test_hdf5.c -o test_hdf5 $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test.hdf5
	@./test_hdf5
	@rm test.hdf5

	$(GCC) test_hdf5_chunked.c -o test_hdf5_chunked $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	rm -f test_chunked.hdf5
	@./test_hdf5_chunked
	@rm test_chunked.hdf5

	$(GCC) test_types.c -o test_types $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_types

	$(GCC) test_transfer.c -o test_transfer $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_transfer

	$(GCC) test_transfer_interp.c -o test_transfer_interp $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_transfer_interp

	$(GCC) test_titles.c -o test_titles $(OBJECTS) $(LIBRARIES) $(CFLAGS) $(INCLUDES)
	@./test_titles
