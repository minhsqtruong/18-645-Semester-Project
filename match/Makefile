COMPILER=gcc
#LDFLAGS= neccessary LD libraries

# Must return match.o
all: match.c
	${COMPILER} -O -c match.c -o match.o

# test contain the main function for match module
test: all tb_match.c
	${COMPILER} -O -o tb_match tb_match.c match.o

clean:
	rm -rf tmp* tb_match *.o
