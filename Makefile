#Compiler options
GCC = gcc

#Libraries
INI_PARSER = parser/minIni.o
STD_LIBRARIES = -lm
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES)

all:
	make minIni
	$(GCC) dexm.c -o dexm $(LIBRARIES)

minIni:
	cd parser && make

check:
	cd tests && make
