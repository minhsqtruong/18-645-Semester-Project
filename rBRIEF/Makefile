COMPILER=gcc
#LDFLAGS= neccessary LD libraries

# Must return rBRIEF.o
all: rBRIEF.c
	${COMPILER} -O -c rBRIEF.c -o rBRIEF.o

# test contain the main function for rBRIEF module
test: all tb_rBRIEF.c
	${COMPILER} -O -o tb_rBRIEF tb_rBRIEF.c rBRIEF.o

clean:
	rm -rf tmp* tb_rBRIEF *.o
