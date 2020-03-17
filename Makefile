#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
FFTW_LIBRARIES = -lfftw3
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES) $(FFTW_LIBRARIES)

OBJECTS = lib/input.o lib/random.o lib/fft.o

all:
	make minIni
	$(GCC) src/input.c -c -o lib/input.o
	$(GCC) src/random.c -c -o lib/random.o
	$(GCC) src/fft.c -c -o lib/fft.o
	$(GCC) src/dexm.c -o dexm $(OBJECTS) $(LIBRARIES)

minIni:
	cd parser && make

check:
	cd tests && make
