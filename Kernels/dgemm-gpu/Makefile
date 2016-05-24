# module swap PrgEnv-cray PrgEnv-gnu; module load cudatoolkit
CC = icc
LIBS = -L/ssoft/nvidia/cuda-7.0/RH6/all/x86_E5v2//lib64/
INCLUDES = -I./ -I${CUDA_INCLUDE}
CFLAGS = -O2 -g0 -openmp -lcublas -lcudart -lcuda
#CFLAGS = -mkl -O2 -g0 -fopenmp -lnvidia-ml -lcublas -lcudart -lcuda
#CFLAGS += -DCHECK
CFLAGS += -DPINNED

MBS = 256
NBS = 32
KBS = 256

#CFLAGS += -DPAPI 
#LIBS   += -L/cm/shared/apps/papi/5.0.1/lib -lpapi
#INCLUDES += -I/cm/shared/apps/papi/5.0.1/include/
#CFLAGS += -DM_BLOCK_SIZE=$(MBS) -DN_BLOCK_SIZE=$(NBS) -DK_BLOCK_SIZE=$(KBS)

#CFLAGS += -DIACA
#INCLUDES += -I/users/fgilles/Projects/Pilatus/iaca-lin32/include/

all:	dgemm-cublas dgemm-buffer dgemm-buffer.v1 dgemm-buffer.v2 dgemm-hpl

dgemm-cublas: dgemm.o dgemm-cublas.o
	$(CC) -o $@ dgemm.o dgemm-cublas.o $(CFLAGS) $(LIBS) $(INCLUDES)
dgemm-buffer: dgemm.o dgemm-buffer.o
	$(CC) -o $@ dgemm.o dgemm-buffer.o $(CFLAGS) $(LIBS) $(INCLUDES)
dgemm-buffer.v1: dgemm.o dgemm-buffer.v1.o
	$(CC) -o $@ dgemm.o dgemm-buffer.v1.o $(CFLAGS) $(LIBS) $(INCLUDES)
dgemm-buffer.v2: dgemm.o dgemm-buffer.v2.o
	$(CC) -o $@ dgemm.o dgemm-buffer.v2.o $(CFLAGS) $(LIBS) $(INCLUDES)
dgemm-hpl: dgemm.o dgemm-hpl.o
	$(CC) -o $@ dgemm.o dgemm-hpl.o $(CFLAGS) $(LIBS) $(INCLUDES)


%.o: %.c
	$(CC) -c $(CFLAGS) $(INCLUDES) $<

clean:
	rm -f *.o dgemm-m dgemm-cublas dgemm-buffer.v? dgemm-hpl

