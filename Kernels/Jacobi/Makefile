CC=gcc
ISPC=ispc
all: lap-avx lap-ispc lap-naive lap-sse lap-sse2 
FLAGS  = -std=gnu99 -g -O3 -march=native -fopenmp
#FLAGS += -DPAPI -I/ssoft/papi/5.4.0/gnu/include/ /ssoft/papi/5.4.0/gnu/lib/libpapi.a
#FLAGS +=  -I/apps/dora/iaca/iaca-lin32/include/

lap-naive: laplacian.o lap-naive.o
	gcc $(FLAGS) -o $@ laplacian.o lap-naive.o 

lap-sse: laplacian.o lap-sse.o
	gcc $(FLAGS) -o $@ laplacian.o lap-sse.o

lap-sse2: laplacian.o lap-sse2.o
	gcc $(FLAGS) -o $@ laplacian.o lap-sse2.o

lap-ispc: lap-ispc.o laplacian.o
	gcc $(FLAGS) -o $@ laplacian.o lap-ispc.o 

lap-avx: laplacian.o lap-avx.o
	gcc $(FLAGS) -o $@ laplacian.o lap-avx.o

%.o: %.c
	$(CC) -c $(FLAGS) $<

%.o: %.ispc
	$(ISPC) -O3 --opt=fast-math  $< -o $@ --target=avx2-i64x4 --cpu=core-avx2

clean:
	rm -f *.lst *.o lap-avx lap-ispc lap-sse lap-sse2 lap-naive

