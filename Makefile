#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES)

OBJECTS = lib/input.o

all:
	make minIni
	$(GCC) src/input.c -c -o lib/input.o
	$(GCC) src/dexm.c -o dexm $(LIBRARIES) $(OBJECTS)

minIni:
	cd parser && make

check:
	cd tests && make
