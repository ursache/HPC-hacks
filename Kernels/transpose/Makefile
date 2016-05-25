kernels ?= avx

config ?= release

CXXFLAGS = -fPIC -fvisibility=hidden -fstrict-aliasing -fopenmp -std=c++11
ifeq "$(config)" "release"
	CXXFLAGS += -march=native -mtune=native -fstrict-aliasing -O4 -march=native -mtune=native -DNDEBUG
else
	CXXFLAGS +=-g -msse4.1
endif

all: xy-transpose xz-transpose

% : test-%.cpp %.o 
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o : %.cpp Makefile
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $@

%.cpp : %.m4
	m4 -D KERNELS=$(kernels) $^ | indent > $@

clean:
	rm -f *.o xy-transpose.cpp xz-transpose.cpp xy-transpose xz-transpose


.PHONY: clean

.PRECIOUS: xz-transpose.cpp xy-transpose.cpp xy-transpose.o xz-transpose.o
