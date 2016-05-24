CC=mpiicc
ISPC=ispc
all: lap-naive 
FLAGS  = -std=gnu99 -g -O3 -march=native -openmp 
FLAGS += -DMPI 
#FLAGS += -DPAPI -I/ssoft/papi/5.4.0/gnu/include/ /ssoft/papi/5.4.0/gnu/lib/libpapi.a
#FLAGS +=  -I/apps/dora/iaca/iaca-lin32/include/

lap-naive: laplacian.o lap-naive.o
	$(CC) $(FLAGS) $^ -o $@ 

lap-sse: laplacian.o lap-sse.o
	$(CC) $(FLAGS) $^ -o $@ 

lap-sse2: laplacian.o lap-sse2.o
	$(CC) $(FLAGS) $^ -o $@ 

lap-ispc: lap-ispc.o laplacian.o tasksys.o
	clang++ $(FLAGS) $^  -o $@ -lpthread
#	gcc $(FLAGS) -o $@ laplacian.o lap-ispc.o tasksys.o

lap-avx: laplacian.o lap-avx.o 
	$(CC) $(FLAGS) $^ -o $@ 
#
lap-avx2: laplacian.o lap-avx2.o 
	$(CC) $(FLAGS) $^ -o $@ 
#
lap-avx3: laplacian.o lap-avx3.o 
	$(CC) $(FLAGS) $^ -o $@ 
#
lap-ispcomp: laplacian.o lap-ispcomp-c.o lap-ispcomp.o 
#lap-ispcomp: laplacian.o lap-ispcomp-c.o lap-ispcomp.o lap-avx.o
#lap-ispcomp: laplacian.o lap-ispcomp-c.o 
	gcc $(FLAGS) $^ -o $@
#
lap-ispcomp2: laplacian.o lap-ispcomp-c.o lap-ispcomp2.o 
	gcc $(FLAGS) $^ -o $@
#
tasksys.o: tasksys.cpp
	clang++ -O2 -c $<

%.o: %.c
	$(CC) -c $(FLAGS) $<
	$(CC) -S $(FLAGS) $< -o $@.s 

%.o: %.ispc
#	$(ISPC) --opt=force-aligned-memory -O3 --opt=fast-math  $< -o $@ --target=avx2-i64x4 --cpu=core-avx2
	$(ISPC) -O3 --opt=fast-math  $< -o $@ --target=avx2-i64x4 --cpu=core-avx2
	$(ISPC) --opt=force-aligned-memory -O3 --opt=fast-math  $< -o $@.s --target=avx2-i64x4 --cpu=core-avx2 --emit-asm

clean:
	rm -f *.s *.lst *.o a.out lap-ispcomp lap-avx lap-avx2 lap-avx3 lap-ispc lap-sse lap-sse2 lap-naive
