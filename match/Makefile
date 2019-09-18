COMPILER=nvcc
#LDFLAGS= neccessary LD libraries

# Must return match.o
all: match.cu
	${COMPILER} -O -c match.cu -o match.o

# test contain the main function for match module
test: all tb_match.cu
	${COMPILER} -O -o tb_match tb_match.cu match.o

clean:
	rm -rf tmp* tb_match *.o
