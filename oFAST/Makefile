COMPILER=gcc
#LDFLAGS= neccessary LD libraries

# Must return oFAST.o
all: oFAST.c
	${COMPILER} -O -c oFAST.c -o oFAST.o

# test contain the main function for oFAST module
test: all tb_oFAST.c
	${COMPILER} -O -o tb_oFAST tb_oFAST.c oFAST.o

clean:
	rm -rf tmp* tb_oFAST *.o
