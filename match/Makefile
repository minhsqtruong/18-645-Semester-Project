COMPILER=/usr/local/cuda/bin/nvcc
#LDFLAGS= neccessary LD libraries

# Must return match.o
all: match.cu
	${COMPILER} -O -c -std=c++11 match.cu -o match.o 

# test contain the main function for match module
test: all tb_match.cu
	${COMPILER} -O -std=c++11 -o tb_match tb_match.cu match.o

clean:
	rm -rf tmp* tb_match *.o
