COMPILER=nvcc
#LDFLAGS= neccessary LD libraries

# Must return oFAST.o
all: oFAST.cu
	${COMPILER} -O -c oFAST.cu -o oFAST.o

# test contain the main function for oFAST module
test: all tb_oFAST.cu
	${COMPILER} -O -o tb_oFAST tb_oFAST.cu oFAST.o

clean:
	rm -rf tmp* tb_oFAST *.o
