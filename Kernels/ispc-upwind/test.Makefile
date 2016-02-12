ifeq "$(real)" "double"
TARGET = avx1.1-i64x4
else
TARGET = avx1.1-i32x8
endif


all: $(real).cpp.test $(real).ispc.test

$(real).%.test: main.cpp upwind3-kernel.%.o
	g++ -DREAL=$(real) -std=c++11 -O3 $^ -o $@

%.ispc.o: %.ispc
	ispc -DREAL=$(real) --target=$(TARGET) -O2 -o $@ $^	

%.cpp.o: %.cpp
	g++ -DREAL=$(real) -O4 -fstrict-aliasing -c $^ -o $@

clean:
	rm -f *.test *.o

PRECIOUS = *.o
.PHONY = clean
