config ?= release
kernels ?= ispc
cpu ?= haswell
isa ?= avx2
masking ?= 32
ISPC ?= /home/diegor/ispc-v1.9.0-linux/ispc
ISPCFLAGS ?=  -O3 --arch=x86-64 --math-lib=fast \
	--opt=disable-assertions --opt=fast-math \
	--cpu=$(cpu) --target=$(isa)-i$(masking)x8 \
	--wno-perf --pic

OBJS += test.o

ifeq "$(kernels)" "ispc"
	OBJS += bitshuffle.ispc.o
else
	OBJS += bitshuffle.o
endif

CXXFLAGS += -fPIC -fvisibility=hidden -fstrict-aliasing -std=c++11 -g

ifeq "$(config)" "release"
	CXXFLAGS += -march=native -mtune=native -fstrict-aliasing -O4 -DNDEBUG -fopenmp -g
else
	CXXFLAGS += -g -fopenmp
endif

test : $(OBJS) 
	$(CXX) $(CXXFLAGS) $^ -o test

-include $(OBJS:.o=.d)

%.ispc.o: %.ispc Makefile
	$(ISPC) $(ISPCFLAGS) $*.ispc -o $@

%.o : %.cpp Makefile
	$(CXX) -c $(CXXFLAGS) $*.cpp -o $*.o
	$(CXX) -MM $(CXXFLAGS) $*.cpp > $*.d

%.ispc: %.m4
	m4 -D KERNELS=ISPC $^ | indent > $@

%.cpp : %.m4
	m4 $^ | indent > $@ 

clean :
	rm -f *.o *.d test bitshuffle.cpp bitshuffle.ispc

.PHONY: all clean
.PRECIOUS: %.cpp %.ispc
