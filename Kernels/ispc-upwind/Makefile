benchmark: all
	./float.ispc.test
	./float.cpp.test
	./double.ispc.test
	./double.cpp.test

all: 
	make -f test.Makefile all real=float
	make -f test.Makefile all real=double

clean:
	rm -f *.test

.PHONY = all clean
