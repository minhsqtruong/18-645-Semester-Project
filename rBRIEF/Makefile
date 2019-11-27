COMPILER=/usr/local/cuda/bin/nvcc
#LDFLAGS= neccessary LD libraries

# Must return rBRIEF.o
all: rBRIEF.cu
	${COMPILER}  -O -c rBRIEF.cu -o rBRIEF.o

# test contain the main function for rBRIEF module
test: all tb_rBRIEF.cu rBRIEF.o
	${COMPILER}   -o tb_rBRIEF tb_rBRIEF.cu rBRIEF.o

clean:
	rm -rf tmp* tb_rBRIEF *.o
