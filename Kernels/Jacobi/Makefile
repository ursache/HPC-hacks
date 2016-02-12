CC=gcc
ISPC=ispc
all: lap-ispcomp lap-avx3 lap-avx2 lap-avx lap-ispc lap-naive lap-sse lap-sse2 
FLAGS  = -std=gnu99 -g -O3 -march=native -fopenmp -lm
#FLAGS += -DPAPI -I/ssoft/papi/5.4.0/gnu/include/ /ssoft/papi/5.4.0/gnu/lib/libpapi.a
#FLAGS +=  -I/apps/dora/iaca/iaca-lin32/include/

lap-naive: laplacian.o lap-naive.o
	gcc $(FLAGS) $^ -o $@ 

lap-sse: laplacian.o lap-sse.o
	gcc $(FLAGS) $^ -o $@ 

lap-sse2: laplacian.o lap-sse2.o
	gcc $(FLAGS) $^ -o $@ 

lap-ispc: lap-ispc.o laplacian.o tasksys.o
	clang++ $(FLAGS) $^  -o $@ -lpthread
#	gcc $(FLAGS) -o $@ laplacian.o lap-ispc.o tasksys.o

lap-avx: laplacian.o lap-avx.o 
	gcc $(FLAGS) $^ -o $@ 
#
lap-avx2: laplacian.o lap-avx2.o 
	gcc $(FLAGS) $^ -o $@ 
#
lap-avx3: laplacian.o lap-avx3.o 
	gcc $(FLAGS) $^ -o $@ 
#
lap-ispcomp: laplacian.o lap-ispcomp-c.o lap-ispcomp.o 
#lap-ispcomp: laplacian.o lap-ispcomp-c.o lap-ispcomp.o lap-avx.o
#lap-ispcomp: laplacian.o lap-ispcomp-c.o 
	gcc $(FLAGS) $^ -o $@
#
tasksys.o: tasksys.cpp
	clang++ -O2 -c $<

%.o: %.c
	$(CC) -c $(FLAGS) $<
	$(CC) -S $(FLAGS) $< -o $@.s 

%.o: %.ispc
	$(ISPC) -O3 --opt=fast-math  $< -o $@ --target=avx2-i64x4 --cpu=core-avx2
	$(ISPC) -O3 --opt=fast-math  $< -o $@.s --target=avx2-i64x4 --cpu=core-avx2 --emit-asm

clean:
	rm -f *.s *.lst *.o a.out lap-ispcomp lap-avx lap-ispc lap-sse lap-sse2 lap-naive
