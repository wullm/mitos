#Compiler options
GCC = gcc

#Libraries
INI_PARSER = ../parser/minIni.o
STD_LIBRARIES = -lm
LIBRARIES = $(INI_PARSER) $(STD_LIBRARIES)

OBJECTS = ../lib/input.o ../lib/random.o

all:
	@#$(GCC) test_minIni.c -o test_minIni $(INI_PARSER)
	@#@./test_minIni

	$(GCC) test_input.c -o test_input $(OBJECTS) $(LIBRARIES)
	@./test_input

	$(GCC) test_random.c -o test_random $(OBJECTS) $(LIBRARIES)
	@./test_random
