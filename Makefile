#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES)
CFLAGS = -Wall

OBJECTS = lib/input.o lib/random.o lib/fft.o

all:
	make minIni
	$(GCC) src/input.c -c -o lib/input.o $(CFLAGS)
	$(GCC) src/random.c -c -o lib/random.o $(CFLAGS)
	$(GCC) src/fft.c -c -o lib/fft.o $(CFLAGS)
	$(GCC) src/grf.c -c -o lib/grf.o $(CFLAGS)
	$(GCC) src/derivatives.c -c -o lib/derivatives.o $(CFLAGS)
	$(GCC) src/dexm.c -o dexm $(OBJECTS) $(LIBRARIES) $(CFLAGS)

minIni:
	cd parser && make

check:
	cd tests && make
