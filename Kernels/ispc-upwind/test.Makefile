ifeq "$(real)" "double"
TARGET = avx1-i64x4,avx1.1-i64x4,avx2-i64x4
else
TARGET = avx1-i32x8,avx1.1-i32x8,avx2-i32x8
endif

OBJECTS = 

all: $(real).cpp.test $(real).ispc.test

$(real).%.test: main.cpp $(real).upwind3-kernel.%.a
	$(CXX) -DREAL=$(real) -O3 $^ -o $@

$(real).upwind3-kernel.ispc.a: upwind3-kernel.ispc
	ispc -DREAL=$(real) --target=$(TARGET) -O2 -o $(real).ispc.upwind3-kernel.o $<
	ar rcs $@ $(real).ispc.upwind3-kernel*.o

$(real).upwind3-kernel.cpp.a: upwind3-kernel.cpp
	$(CXX) -DREAL=$(real) -O4 -fstrict-aliasing -c $^ -o $(real).cpp.upwind3-kernel.o
	ar rcs $@ $(real).cpp.upwind3-kernel.o

clean:
	rm -f *.test *.o *.a

PRECIOUS = *.o
.PHONY = clean
