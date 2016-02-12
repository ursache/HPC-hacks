#CC=mpiicc

include make.inc

all: laplacian laplacian_naive laplacian_sse laplacian_sse_v2 laplacian_avx laplacian_avx_v2

%.o: %.c
	$(CC) -c $< $(INCLUDES) $(CFLAGS)
	$(CC) -S $< $(INCLUDES) $(CFLAGS)

laplacian_vec:  laplacian_vec.o
	$(CC) $(CFLAGS) -o $@ $<
#
laplacian_avx:  laplacian_avx.o
	$(CC) $(CFLAGS) -o $@ $<
#
laplacian_sse:  laplacian_sse.o
	$(CC) $(CFLAGS) -o $@ $<
#
laplacian_sse_v2:  laplacian_sse_v2.o
	$(CC) $(CFLAGS) -o $@ $<

laplacian_avx_v2:  laplacian_avx_v2.o
	$(CC) $(CFLAGS) -o $@ $<

laplacian_naive: laplacian_naive.o
	$(CC) $(CFLAGS) -o $@ $<

laplacian:  laplacian.o
	$(CC) $(CFLAGS) -o $@ $<


clean:
	rm *.o *.s *.optrpt laplacian laplacian_sse laplacian_sse_v2 laplacian_avx laplacian_avx_v2
