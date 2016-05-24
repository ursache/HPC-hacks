CC = gcc
CXX = g++
#LIBS = /users/fgilles/Projects/OpenBlas/libopenblas.a 
INCLUDES = 

CFLAGS = -g -fopenmp /home/foureste/Projects/OpenBLAS-0.2.14/libopenblas.a -O -march=core-avx2 -fstrict-aliasing 

MBS = 256
NBS = 32
KBS = 256

#all:	ddot-opt ddot-splat ddot-naive ddot-transpose ddot-block ddot-copyblock ddot-unroll  
all:	ddot-naive #ddot-fma #ddot-transpose ddot-copyblock ddot-unroll ddot-arm64 ddot-arm64-8x4 
#all:	ddot-fma ddot-arm64-8x4 arm-pp ddot-arm64 ddot-unroll ddot-ref ddot-copyblock ddot-block ddot-naive ddot-block ddot-transpose neon 


neon: neon.o
	$(CC) -o $@ neon.o $(CFLAGS)
ddot-naive: ddot.o ddot-naive.o
	$(CC) -o $@ ddot.o ddot-naive.o $(CFLAGS) $(LIBS) $(INCLUDES)
ddot-neon: ddot.o ddot-neon.o
	$(CC) -o $@ ddot.o ddot-neon.o $(CFLAGS) $(LIBS)
#
%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $<
#
%.o: %.cpp
	$(CXX) -c $(CPPFLAGS) $(INCLUDES) $<
#
clean:
	rm -f ddot-fma ddot-naive *.o 

