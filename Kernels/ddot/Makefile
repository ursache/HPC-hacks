CC = icc
CXX = icpc
#LIBS = /users/fgilles/Projects/OpenBlas/libopenblas.a 
INCLUDES = 

CFLAGS += -g -mkl -O -march=native -fstrict-aliasing 
CFLAGS += -DCHECK 
CFLAGS += -qopenmp 
#CFLAGS += -qopt_report1 


#all:	ddot-opt ddot-splat ddot-naive ddot-transpose ddot-block ddot-copyblock ddot-unroll  
all:	ddot-naive 
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

