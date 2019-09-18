# Overall Makefile that compile the entire project and create shared lib
# Don't modify this file, each module has it s own Makefile. Those can be
# modified to taste, depending on the module

SUBDIR = match oFAST rBRIEF
EXEC = match/match.o oFAST/oFAST.o rBRIEF/rBRIEF.o
COMPILER=nvcc
#LDFLAGS= neccessary LD libraries
all:
	for i in ${SUBDIR}; do \
		(echo $$i; cd $$i; make all;); \
	done
	${COMPILER} -O -fPIC -c orb.c -o orb.o
	${COMPILER} -O -o tb_orb tb_orb.c orb.o ${EXEC}

lib: all
	${COMPILER} orb -shared -o ./lib/liborb.so

clean:
	for i in ${SUBDIR}; do \
		(echo $$i; cd $$i; make clean;); \
	done
	rm -rf tb_orb tmp* a.out *.o
