include ./make.inc
#
all:    split_v1 split_v2 split_v3 split_v4 split_v5 split_v6 split_v7 split_v8
#
#split-hp: split.o dgemm_staged.o dgemm_split.o dgemm_hp-staged.o
split_v1: split.o dgemm_v1.o
	$(CC) -o $@ split.o dgemm_v1.o $(LIBS) $(OPTS)

split_v2: split.o dgemm_v2.o
	$(CC) -o $@ split.o dgemm_v2.o $(LIBS) $(OPTS)
#
split_v3: split.o dgemm_v3.o
	$(CC) -o $@ split.o dgemm_v3.o $(LIBS) $(OPTS)

split_v4: split.o dgemm_v4.o
	$(CC) -o $@ split.o dgemm_v4.o $(LIBS) $(OPTS)

split_v5: split.o dgemm_v5.o
	$(CC) -o $@ split.o dgemm_v5.o $(LIBS) $(OPTS)

split_v6: split.o dgemm_v6.o
	$(CC) -o $@ split.o dgemm_v6.o $(LIBS) $(OPTS)

split_v7: split.o dgemm_v7.o add.o
	$(CC) -o $@ split.o dgemm_v7.o add.o $(LIBS) $(OPTS)

split_v8: split.o dgemm_v8.o
	$(CC) -o $@ split.o dgemm_v8.o $(LIBS) $(OPTS)

#
%.o : %.cu
	$(NVCC) $(NVOPTS) -c $< -o $*.o 

%.o : %.c
	$(CC) -c $< -o $*.o $(OPTS) $(INC) $(DEFINES) 

#
#lib:
#	ar rvs libdgemm.a dgemm_staged.o dgemm_split.o dgemm_hp.o
#
clean:
	rm -f *.o *.optrpt split_v1 split_v2 split_v3 split_v4 split_v5 split_v6 split_v7 split_v8 


